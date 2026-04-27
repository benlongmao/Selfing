import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import decode_header
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Inbox/sent: how many messages per fetch (aligned with chat_service / tool_router default)
DEFAULT_EMAIL_FETCH_LIMIT = 5


class EmailTool:
    """
    邮件处理工具 (适配 163 邮箱)
    """
    def __init__(self):
        # Load SMTP/IMAP settings from env
        self.email_address = os.getenv("EMAIL_USER")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.163.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "465"))
        self.imap_server = os.getenv("IMAP_SERVER", "imap.163.com")
        
        # Warn if credentials missing
        if not self.email_address or not self.email_password:
            logger.warning("EmailTool 未配置: 缺少 EMAIL_USER 或 EMAIL_PASSWORD")

    def _is_configured(self) -> bool:
        return bool(self.email_address and self.email_password)

    def send_email(self, to_address: str, subject: str, content: str, attachments: Optional[List[str]] = None) -> str:
        """
        发送邮件（支持附件）
        
        :param to_address: 收件人邮箱
        :param subject: 邮件标题
        :param content: 邮件正文
        :param attachments: 附件文件路径列表（可选）
        
        [v2.4 2026-01-14] 新增附件支持，让 Agent 能发送文件
        """
        if not self._is_configured():
            return "❌ 邮件功能未配置，请检查环境变量。"

        try:
            # Optional attachments
            attachment_info = []
            if attachments:
                for file_path in attachments:
                    # Expand ~ etc.
                    file_path = os.path.expanduser(file_path)
                    
                    # Resolve relative paths inside sandbox search roots
                    if not os.path.isabs(file_path):
                        # Repo root = parent of backend/
                        current_file = os.path.abspath(__file__)  # this file
                        tools_dir = os.path.dirname(current_file)  # backend/tools/
                        backend_dir = os.path.dirname(tools_dir)  # backend/
                        project_root = os.path.dirname(backend_dir)  # repo root
                        
                        # [2026-02-28] Attachment lookup confined to workspace/sandbox
                        sandbox_root = os.path.join(project_root, "workspace/sandbox")
                        search_paths = [
                            os.path.join(sandbox_root, file_path),
                            os.path.join(sandbox_root, "code", file_path),
                            os.path.join(sandbox_root, "docs", file_path),
                        ]
                        
                        # Probe candidate roots
                        found = False
                        for candidate_path in search_paths:
                            if os.path.exists(candidate_path):
                                file_path = candidate_path
                                found = True
                                break
                        
                        if not found:
                            # Not found under sandbox roots
                            searched_locations = "\n  - ".join([
                                os.path.relpath(p, project_root) for p in search_paths
                            ])
                            return f"❌ 附件文件未找到: {os.path.basename(file_path)}\n已搜索位置:\n  - {searched_locations}"
                    
                    # Ensure resolved path exists
                    if not os.path.exists(file_path):
                        return f"❌ 附件文件不存在: {file_path}"
                    
                    # 50MB per-attachment cap
                    file_size = os.path.getsize(file_path)
                    if file_size > 50 * 1024 * 1024:  # 50MB
                        size_mb = file_size / 1024 / 1024
                        return f"❌ 附件文件过大: {os.path.basename(file_path)} ({size_mb:.1f}MB)，超过50MB限制"
                    
                    attachment_info.append((file_path, os.path.basename(file_path), file_size))
            
            # Build MIME message
            if attachments:
                # Multipart when attachments present
                msg = MIMEMultipart()
                msg['Subject'] = subject
                msg['From'] = self.email_address
                msg['To'] = to_address
                
                # Body part
                msg.attach(MIMEText(content, 'plain', 'utf-8'))
                
                # Attach each file
                for file_path, filename, file_size in attachment_info:
                    try:
                        with open(file_path, 'rb') as f:
                            # MIME attachment part
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            
                            # Content-Disposition header
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename="{filename}"'
                            )
                            msg.attach(part)
                            
                        logger.info(f"附件已添加: {filename} ({file_size} bytes)")
                    except Exception as e:
                        logger.error(f"添加附件失败 {filename}: {e}")
                        return f"❌ 添加附件失败: {filename} - {str(e)}"
            else:
                # Plain MIMEText when no attachments
                msg = MIMEText(content, 'plain', 'utf-8')
                msg['Subject'] = subject
                msg['From'] = self.email_address
                msg['To'] = to_address

            # SMTP_SSL
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            # server.set_debuglevel(1)  # enable when debugging SMTP
            
            # Auth + send_message
            server.login(self.email_address, self.email_password)
            server.send_message(msg)
            server.quit()
            
            # User-facing success string
            if attachments:
                attachment_names = [info[1] for info in attachment_info]
                logger.info(f"邮件已发送至 {to_address}，包含 {len(attachment_names)} 个附件")
                return f"✅ 邮件已成功发送给 {to_address}\n📎 附件: {', '.join(attachment_names)}"
            else:
                logger.info(f"邮件已发送至 {to_address}")
                return f"✅ 邮件已成功发送给 {to_address}"
            
        except smtplib.SMTPAuthenticationError:
            return "❌ 发送失败：认证错误。请检查邮箱账号或授权码是否正确。"
        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
            return f"❌ 发送失败: {str(e)}"

    def _is_163_mailbox(self) -> bool:
        """判断是否为网易 163/126 邮箱（需发送 IMAP ID 命令）"""
        s = (self.imap_server or "").lower()
        return "163.com" in s or "126.com" in s

    def _parse_email_message(
        self,
        msg: email.message.Message,
        e_id,
        *,
        is_sent_folder: bool = False,
    ) -> str:
        """解析单封邮件为展示字符串。"""
        return self._parse_email_message_impl(msg, e_id, is_sent_folder=is_sent_folder)

    def _parse_email_message_impl(
        self, msg: email.message.Message, e_id, *, is_sent_folder: bool = False
    ) -> str:
        """解析单封邮件为展示字符串（内部实现）"""
        # Decode Subject header
        subject_raw = msg["Subject"]
        if subject_raw:
            decoded_list = decode_header(subject_raw)
            subject_parts = []
            for content, encoding in decoded_list:
                if isinstance(content, bytes):
                    subject_parts.append(content.decode(encoding if encoding else "utf-8"))
                else:
                    subject_parts.append(str(content))
            subject = "".join(subject_parts)
        else:
            subject = "(无标题)"
        from_ = msg.get("From")
        body = "(无法解析正文)"
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            body = payload.decode(charset, errors="replace")
                            break
                    except Exception as e:
                        logger.warning(f"解析邮件正文出错: {e}")
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
            except Exception:
                pass
        body_preview = body[:300].replace("\n", " ") + ("..." if len(body) > 300 else "")
        if is_sent_folder:
            to_ = msg.get("To") or "(无收件人)"
            return (
                f"📤 [邮件 ID: {e_id}]\n"
                f"   收件人: {to_}\n"
                f"   发件人: {from_}\n"
                f"   标题: {subject}\n"
                f"   摘要: {body_preview}\n"
            )
        return f"📩 [邮件 ID: {e_id}]\n   发件人: {from_}\n   标题: {subject}\n   摘要: {body_preview}\n"

    def check_unread_emails(
        self,
        limit: int = DEFAULT_EMAIL_FETCH_LIMIT,
        only_unread: bool = False,
        folder: str = "inbox",
    ) -> str:
        """
        查看邮件（收件箱或已发送等文件夹）。
        [2026-03] 163/126 邮箱需在 login 后、select 前发送 IMAP ID 命令，否则会返回 "Unsafe Login"/NO
        使用 imapclient 的 id_() 解决该问题
        :param limit: 获取最新的几封
        :param only_unread: True 时仅 UNSEEN；False 时取文件夹内最近若干封（含已读），对应「看收件箱」而不仅是「未读」
        :param folder: \"inbox\" 收件箱；\"sent\" 已发送（需 IMAP 可读该文件夹；可设环境变量 IMAP_SENT_FOLDER）
        """
        if not self._is_configured():
            return "❌ 邮件功能未配置。"

        fk = (folder or "inbox").strip().lower()
        if fk not in ("inbox", "sent"):
            return f"❌ 不支持的 folder={folder!r}，请使用 inbox 或 sent。"

        if self._is_163_mailbox():
            return self._check_unread_emails_imapclient(
                limit, only_unread=only_unread, folder_kind=fk
            )
        return self._check_unread_emails_imaplib(
            limit, only_unread=only_unread, folder_kind=fk
        )

    def _detect_sent_folder_imapclient(self, client) -> str:
        """解析「已发送」文件夹名称（网易/Gmail/通用常见名 + list_folders 探测）。"""
        explicit = (os.getenv("IMAP_SENT_FOLDER") or "").strip()
        if explicit:
            return explicit
        candidates = [
            "已发送",
            "Sent Messages",
            "Sent",
            "[Gmail]/Sent Mail",
            "INBOX.Sent",
            "Sent Items",
        ]
        existing: List[str] = []
        try:
            for item in client.list_folders():
                # (flags, delimiter, name) or legacy tuple shapes
                name = item[-1]
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                existing.append(name)
        except Exception as e:
            logger.warning(f"IMAP list_folders 失败: {e}")

        for c in candidates:
            if c in existing:
                return c
        for name in existing:
            nl = name.lower()
            if "sent" in nl or "已发" in name:
                return name
        # Fallback name; select() will error if wrong
        return candidates[0]

    def _detect_sent_folder_imaplib(self, mail: imaplib.IMAP4_SSL) -> str:
        explicit = (os.getenv("IMAP_SENT_FOLDER") or "").strip()
        if explicit:
            return explicit
        candidates = [
            "已发送",
            "Sent Messages",
            "Sent",
            "[Gmail]/Sent Mail",
            "INBOX.Sent",
            "Sent Items",
        ]
        try:
            status, raw = mail.list()
            if status == "OK" and raw:
                existing = []
                for line in raw:
                    if not line:
                        continue
                    # e.g. b'(\HasNoChildren) "/" "INBOX"'
                    parts = line.decode("utf-8", errors="replace").rsplit('"', 2)
                    if len(parts) >= 2:
                        existing.append(parts[-2])
                for c in candidates:
                    if c in existing:
                        return c
                for name in existing:
                    nl = name.lower()
                    if "sent" in nl or "已发" in name:
                        return name
        except Exception as e:
            logger.warning(f"IMAP list 解析失败: {e}")
        return candidates[0]

    def _check_unread_emails_imapclient(
        self, limit: int, only_unread: bool = False, folder_kind: str = "inbox"
    ) -> str:
        """使用 imapclient 收取邮件（163/126 专用，支持 IMAP ID）"""
        try:
            try:
                from imapclient import IMAPClient
            except ImportError:
                return (
                    "❌ 163 邮箱需要 imapclient 库。请运行: pip install imapclient\n"
                    "（163 要求发送 IMAP ID 命令，标准库 imaplib 不支持）"
                )

            with IMAPClient(self.imap_server, ssl=True, port=993) as client:
                client.login(self.email_address, self.email_password)
                # NetEase 163/126: IMAP ID after login, before SELECT ("Unsafe Login" otherwise)
                # Vendor string mimics common MUAs whitelisted by provider
                client.id_({
                    "name": "IMAPClient",
                    "version": "2.1.0",
                    "vendor": "Mozilla",
                    "contact": self.email_address or "user@localhost",
                })
                is_sent = folder_kind == "sent"
                if is_sent:
                    mbox = self._detect_sent_folder_imapclient(client)
                    logger.info(f"IMAP 已发送文件夹: {mbox!r}")
                else:
                    mbox = "INBOX"
                try:
                    client.select_folder(mbox)
                except Exception as sel_err:
                    return (
                        f"❌ 无法打开文件夹 {mbox!r}（{'已发送' if is_sent else '收件箱'}）。"
                        f"错误: {sel_err}\n"
                        "提示：可在 .env 设置 IMAP_SENT_FOLDER=服务器上准确的文件夹名"
                        "（网页邮箱设置里可见 IMAP 文件夹名）。"
                    )
                criteria = ["UNSEEN"] if only_unread else ["ALL"]
                messages = client.search(criteria)
                if not messages:
                    if is_sent:
                        if only_unread:
                            return "📭 已发送文件夹中没有符合「未读」条件的邮件（通常发件箱无未读）。"
                        return "📭 已发送文件夹中暂无邮件。"
                    if only_unread:
                        return "📭 收件箱没有未读邮件。"
                    return "📭 收件箱中暂无邮件。"
                latest_ids = messages[-limit:]
                scope = "未读" if only_unread else "最近"
                loc = "已发送" if is_sent else "收件箱"
                logger.info(f"IMAP {loc}({scope}): 匹配 {len(messages)} 封，读取最新 {len(latest_ids)} 封")
                result_list = []
                for msg_id, data in client.fetch(latest_ids, ["RFC822"]).items():
                    rfc822 = data.get(b"RFC822")
                    if rfc822:
                        msg = email.message_from_bytes(rfc822)
                        result_list.append(
                            self._parse_email_message(
                                msg, msg_id, is_sent_folder=is_sent
                            )
                        )
                if is_sent:
                    header = (
                        f"已发送文件夹最近 {len(result_list)} 封（未读筛选）：\n"
                        if only_unread
                        else f"已发送文件夹最近 {len(result_list)} 封：\n"
                    )
                else:
                    header = (
                        f"找到 {len(result_list)} 封未读邮件：\n"
                        if only_unread
                        else f"收件箱最近 {len(result_list)} 封邮件（含已读/未读）：\n"
                    )
                return header + "\n".join(result_list)
        except Exception as e:
            return self._format_imap_error(e)

    def _check_unread_emails_imaplib(
        self, limit: int, only_unread: bool = False, folder_kind: str = "inbox"
    ) -> str:
        """使用 imaplib 收取邮件（非 163 邮箱）"""
        try:
            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.email_address, self.email_password)
            is_sent = folder_kind == "sent"
            if is_sent:
                mbox = self._detect_sent_folder_imaplib(mail)
                logger.info(f"IMAP(imaplib) 已发送文件夹: {mbox!r}")
            else:
                mbox = "INBOX"
            status, _ = mail.select(mbox)
            if status != "OK":
                return (
                    f"❌ 无法打开文件夹 {mbox!r}。请检查 IMAP_SENT_FOLDER 或服务器文件夹名。"
                    f" 状态: {status}"
                )
            crit = "UNSEEN" if only_unread else "ALL"
            status, messages = mail.search(None, crit)
            if status != "OK":
                return "无法搜索邮件。"
            email_ids = messages[0].split()
            if not email_ids:
                if is_sent:
                    if only_unread:
                        return "📭 已发送文件夹中没有符合「未读」条件的邮件（通常发件箱无未读）。"
                    return "📭 已发送文件夹中暂无邮件。"
                if only_unread:
                    return "📭 收件箱没有未读邮件。"
                return "📭 收件箱中暂无邮件。"
            latest_email_ids = email_ids[-limit:]
            result_list = []
            for e_id in latest_email_ids:
                _, msg_data = mail.fetch(e_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        result_list.append(
                            self._parse_email_message(
                                msg, e_id.decode(), is_sent_folder=is_sent
                            )
                        )
                        break
            mail.close()
            mail.logout()
            if is_sent:
                header = (
                    f"已发送文件夹最近 {len(result_list)} 封（未读筛选）：\n"
                    if only_unread
                    else f"已发送文件夹最近 {len(result_list)} 封：\n"
                )
            else:
                header = (
                    f"找到 {len(result_list)} 封未读邮件：\n"
                    if only_unread
                    else f"收件箱最近 {len(result_list)} 封邮件（含已读/未读）：\n"
                )
            return header + "\n".join(result_list)
        except Exception as e:
            return self._format_imap_error(e)

    def _format_select_error(self, status: str) -> str:
        """格式化 select 失败时的错误信息，含 163 排查指引"""
        hint = ""
        if "163" in (self.imap_server or "") or "126" in (self.imap_server or ""):
            hint = (
                "\n\n163/126 邮箱提示：请确保使用授权码（非登录密码）、"
                "已开启 IMAP 服务。若仍失败，可在网页版 设置→POP3/SMTP/IMAP 中验证身份。"
            )
        return f"❌ 无法打开收件箱。服务器返回: {status}{hint}"

    def _format_imap_error(self, e: Exception) -> str:
        """统一格式化 IMAP 相关错误"""
        if isinstance(e, imaplib.IMAP4.error):
            msg = ""
            try:
                if getattr(e, "args", None):
                    first = e.args[0]
                    if isinstance(first, (bytes, bytearray)):
                        msg = first.decode("utf-8", errors="replace")
                    else:
                        msg = str(first)
                else:
                    msg = str(e)
            except Exception:
                msg = str(e)
            try:
                s = msg.strip()
                if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
                    import ast
                    b = ast.literal_eval(s)
                    if isinstance(b, (bytes, bytearray)):
                        msg = b.decode("utf-8", errors="replace")
            except Exception:
                pass
            return f"❌ 邮箱登录失败: {msg}。请检查授权码。"
        # imapclient may raise IMAPClientError subclasses
        err_msg = str(e)
        if "Unsafe Login" in err_msg or "NO" in err_msg:
            return (
                f"❌ 收取失败: {err_msg}\n\n"
                "163 邮箱提示：请使用授权码（非登录密码）、确保 IMAP 已开启。"
                "若仍失败，登录 mail.163.com → 设置 → POP3/SMTP/IMAP 验证身份。"
            )
        logger.error(f"收取邮件时发生错误: {e}")
        return f"❌ 收取失败: {err_msg}"
