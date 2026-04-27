#!/usr/bin/env python3
"""
Human-in-the-loop approval system.

Sensitive actions require explicit human confirmation before execution.

Responsibilities:
1. Classify sensitive operations
2. Create approval requests
3. Poll / surface approval status
4. Run actions after approval
"""
import sqlite3
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"


class RiskLevel(Enum):
    LOW = "low"  # auto-approve when policy allows
    MEDIUM = "medium"  # needs confirmation
    HIGH = "high"  # detailed review
    CRITICAL = "critical"  # multi-step confirmation


# action_name -> (base risk, human-readable reason for the model / UI)
SENSITIVE_ACTIONS = {
    "send_email": (RiskLevel.MEDIUM, "Send email to an external address"),
    "execute_python": (RiskLevel.MEDIUM, "Execute Python code"),
    "delete_file": (RiskLevel.HIGH, "Delete a file"),
    "goal_update_status": (RiskLevel.LOW, "Update goal status"),
    "write_file": (RiskLevel.LOW, "Write a file"),
    "agent_memory_record": (RiskLevel.LOW, "Append sandbox/agent_memory run log"),
    "agent_memory_sync": (RiskLevel.LOW, "Refresh sandbox/agent_memory snapshot"),
    "rename_file": (RiskLevel.LOW, "Rename or move a file inside the workspace"),
}


class ApprovalSystem:
    """SQLite-backed approval queue for gated tool actions."""

    def __init__(self, db_path: str = "data.db", auto_approve_low: bool = True):
        self.db_path = db_path
        self.auto_approve_low = auto_approve_low
        self.approval_timeout_minutes = 60
        self._ensure_table()

    def _ensure_table(self):
        """Create the approval_requests table if missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS approval_requests (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        action_name TEXT NOT NULL,
                        action_args TEXT,
                        risk_level TEXT NOT NULL,
                        description TEXT,
                        status TEXT DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        expires_at TEXT,
                        decided_at TEXT,
                        decided_by TEXT,
                        rejection_reason TEXT,
                        execution_result TEXT
                    )
                """)
                conn.commit()
                logger.info("Approval system table ensured")
        except Exception as e:
            logger.error(f"Failed to ensure approval table: {e}")

    def check_need_approval(self, action_name: str, args: Dict) -> Dict[str, Any]:
        """Return whether this call must wait for human approval.

        Returns:
            {
                "need_approval": bool,
                "risk_level": str,
                "reason": str
            }
        """
        if action_name not in SENSITIVE_ACTIONS:
            return {
                "need_approval": False,
                "risk_level": "none",
                "reason": "Action is not on the sensitive list",
            }

        risk_level, description = SENSITIVE_ACTIONS[action_name]

        if risk_level == RiskLevel.LOW and self.auto_approve_low:
            return {
                "need_approval": False,
                "risk_level": risk_level.value,
                "reason": "Low-risk action auto-approved by policy",
            }

        adjusted_risk = self._assess_risk(action_name, args, risk_level)

        return {
            "need_approval": adjusted_risk != RiskLevel.LOW or not self.auto_approve_low,
            "risk_level": adjusted_risk.value,
            "reason": description,
        }

    def _assess_risk(self, action_name: str, args: Dict, base_risk: RiskLevel) -> RiskLevel:
        """Raise risk based on concrete arguments."""
        if action_name == "send_email":
            to_address = args.get("to_address", "")
            if not to_address.endswith(("@internal.com", "@example.com")):
                return RiskLevel.HIGH

        if action_name == "execute_python":
            code = args.get("code", "")
            if "open(" in code or "write" in code:
                return RiskLevel.HIGH

        if action_name == "delete_file":
            return RiskLevel.HIGH

        return base_risk

    def request_approval(
        self,
        session_id: str,
        action_name: str,
        action_args: Dict,
        description: str = "",
    ) -> Dict[str, Any]:
        """Insert a pending approval row unless auto-approved.

        Returns:
            {
                "success": bool,
                "approval_id": str,
                "status": str,
                "message": str
            }
        """
        try:
            approval_id = f"apr-{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(minutes=self.approval_timeout_minutes)

            risk_info = self.check_need_approval(action_name, action_args)
            risk_level = risk_info["risk_level"]

            if not risk_info["need_approval"]:
                return {
                    "success": True,
                    "approval_id": approval_id,
                    "status": ApprovalStatus.APPROVED.value,
                    "message": "Auto-approved (low-risk action)",
                    "auto_approved": True,
                }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO approval_requests
                    (id, session_id, action_name, action_args, risk_level,
                     description, status, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        approval_id,
                        session_id,
                        action_name,
                        json.dumps(action_args, ensure_ascii=False),
                        risk_level,
                        description or risk_info["reason"],
                        ApprovalStatus.PENDING.value,
                        now.isoformat(),
                        expires_at.isoformat(),
                    ),
                )
                conn.commit()

            logger.info(f"Approval request created: {approval_id} for {action_name}")

            return {
                "success": True,
                "approval_id": approval_id,
                "status": ApprovalStatus.PENDING.value,
                "message": f"Approval request created; awaiting human confirmation. Risk: {risk_level}",
                "expires_at": expires_at.isoformat(),
                "risk_level": risk_level,
            }

        except Exception as e:
            logger.error(f"Failed to create approval request: {e}")
            return {"success": False, "error": str(e)}

    def get_approval_status(self, approval_id: str) -> Dict[str, Any]:
        """Fetch a single approval row and refresh expiry if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT * FROM approval_requests WHERE id = ?",
                    (approval_id,),
                )
                row = cur.fetchone()

                if not row:
                    return {"success": False, "error": "Approval request not found"}

                result = dict(row)
                result["action_args"] = json.loads(result["action_args"]) if result["action_args"] else {}

                if result["status"] == ApprovalStatus.PENDING.value:
                    expires_at = datetime.fromisoformat(result["expires_at"])
                    if datetime.now(timezone.utc) > expires_at:
                        self._update_status(approval_id, ApprovalStatus.EXPIRED)
                        result["status"] = ApprovalStatus.EXPIRED.value

                return {"success": True, "approval": result}

        except Exception as e:
            logger.error(f"Failed to get approval status: {e}")
            return {"success": False, "error": str(e)}

    def get_pending_approvals(self, session_id: Optional[str] = None) -> List[Dict]:
        """List pending approvals, optionally scoped to one session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                if session_id:
                    cur = conn.execute(
                        """
                        SELECT * FROM approval_requests
                        WHERE status = ? AND session_id = ?
                        ORDER BY created_at DESC
                        """,
                        (ApprovalStatus.PENDING.value, session_id),
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT * FROM approval_requests
                        WHERE status = ?
                        ORDER BY created_at DESC
                        """,
                        (ApprovalStatus.PENDING.value,),
                    )

                results = []
                now = datetime.now(timezone.utc)

                for row in cur.fetchall():
                    item = dict(row)
                    item["action_args"] = json.loads(item["action_args"]) if item["action_args"] else {}

                    expires_at = datetime.fromisoformat(item["expires_at"])
                    if now > expires_at:
                        self._update_status(item["id"], ApprovalStatus.EXPIRED)
                        continue

                    remaining = (expires_at - now).total_seconds()
                    item["remaining_seconds"] = int(remaining)

                    results.append(item)

                return results

        except Exception as e:
            logger.error(f"Failed to get pending approvals: {e}")
            return []

    def approve(self, approval_id: str, approved_by: str = "human") -> Dict[str, Any]:
        """Mark a request approved."""
        return self._decide(approval_id, ApprovalStatus.APPROVED, approved_by)

    def reject(self, approval_id: str, reason: str = "", rejected_by: str = "human") -> Dict[str, Any]:
        """Mark a request rejected."""
        return self._decide(approval_id, ApprovalStatus.REJECTED, rejected_by, reason)

    def _decide(
        self,
        approval_id: str,
        status: ApprovalStatus,
        decided_by: str,
        rejection_reason: str = "",
    ) -> Dict[str, Any]:
        """Persist approve / reject decision."""
        try:
            now = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT status FROM approval_requests WHERE id = ?",
                    (approval_id,),
                )
                row = cur.fetchone()

                if not row:
                    return {"success": False, "error": "Approval request not found"}

                if row[0] != ApprovalStatus.PENDING.value:
                    return {"success": False, "error": f"Request already finalized; status={row[0]}"}

                conn.execute(
                    """
                    UPDATE approval_requests
                    SET status = ?, decided_at = ?, decided_by = ?, rejection_reason = ?
                    WHERE id = ?
                    """,
                    (status.value, now, decided_by, rejection_reason, approval_id),
                )
                conn.commit()

            verb = "approved" if status == ApprovalStatus.APPROVED else "rejected"
            logger.info(f"Approval {approval_id} {verb} by {decided_by}")

            return {
                "success": True,
                "status": status.value,
                "message": f"Request {verb}",
            }

        except Exception as e:
            logger.error(f"Failed to process approval decision: {e}")
            return {"success": False, "error": str(e)}

    def _update_status(self, approval_id: str, status: ApprovalStatus):
        """Force-update status (e.g. expiry sweep)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE approval_requests SET status = ? WHERE id = ?",
                    (status.value, approval_id),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update approval status: {e}")

    def mark_executed(self, approval_id: str, result: Dict) -> Dict[str, Any]:
        """Persist execution outcome after an approved callback runs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE approval_requests
                    SET status = ?, execution_result = ?
                    WHERE id = ?
                    """,
                    (
                        ApprovalStatus.EXECUTED.value,
                        json.dumps(result, ensure_ascii=False),
                        approval_id,
                    ),
                )
                conn.commit()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tool definitions for approval helpers."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "approval_list_pending",
                    "description": "List pending human-approval requests for this workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "approval_check_status",
                    "description": "Fetch status for a specific approval request by id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "approval_id": {
                                "type": "string",
                                "description": "Approval request id (apr-…)",
                            }
                        },
                        "required": ["approval_id"],
                    },
                },
            },
        ]

    def route_tool_call(self, tool_name: str, args: Dict, session_id: str) -> Dict:
        """Dispatch model-facing approval tools."""
        if tool_name == "approval_list_pending":
            pending = self.get_pending_approvals(session_id)
            return {
                "success": True,
                "pending_count": len(pending),
                "approvals": pending,
            }
        if tool_name == "approval_check_status":
            return self.get_approval_status(args.get("approval_id", ""))
        return {"error": f"Unknown tool: {tool_name}"}


class ApprovalMiddleware:
    """Wrap synchronous tool executors with approval gating."""

    def __init__(self, approval_system: ApprovalSystem):
        self.approval_system = approval_system
        self._pending_callbacks: Dict[str, Callable] = {}

    def wrap_tool_call(
        self,
        action_name: str,
        action_args: Dict,
        session_id: str,
        executor: Callable,
    ) -> Dict[str, Any]:
        """Run ``executor`` immediately, enqueue approval, or return await payload.

        Args:
            action_name: Registered tool / action key
            action_args: JSON-serializable arguments
            session_id: Owning chat session
            executor: Zero-arg callable that performs the side effect

        Returns:
            Tool result dict, or an ``awaiting_approval`` envelope
        """
        check_result = self.approval_system.check_need_approval(action_name, action_args)

        if not check_result["need_approval"]:
            return executor()

        approval_result = self.approval_system.request_approval(
            session_id=session_id,
            action_name=action_name,
            action_args=action_args,
            description=f"Execute {action_name}",
        )

        if approval_result.get("auto_approved"):
            return executor()

        approval_id = approval_result.get("approval_id")
        if approval_id:
            self._pending_callbacks[approval_id] = executor

        return {
            "awaiting_approval": True,
            "approval_id": approval_id,
            "message": "Action requires human approval. Confirm in the UI, then retry.",
            "risk_level": check_result["risk_level"],
            "expires_at": approval_result.get("expires_at"),
        }

    def execute_approved(self, approval_id: str) -> Dict[str, Any]:
        """Invoke the stored callback after verifying APPROVED status."""
        status_result = self.approval_system.get_approval_status(approval_id)
        if not status_result.get("success"):
            return status_result

        approval = status_result["approval"]

        if approval["status"] != ApprovalStatus.APPROVED.value:
            return {
                "success": False,
                "error": f"Request not approved; current status: {approval['status']}",
            }

        executor = self._pending_callbacks.get(approval_id)
        if not executor:
            return {
                "success": False,
                "error": "Execution callback missing or expired; create a new approval request",
            }

        try:
            result = executor()
            self.approval_system.mark_executed(approval_id, result)
            del self._pending_callbacks[approval_id]
            return result

        except Exception as e:
            logger.error(f"Failed to execute approved action: {e}")
            return {"success": False, "error": str(e)}
