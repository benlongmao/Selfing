@echo off
chcp 65001 >nul

echo ========================================
echo   Agent Browser Control - Chrome Setup
echo ========================================
echo.

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Not running as administrator!
    echo [WARNING] Port forwarding may not work.
    echo [TIP] Right-click this file and select "Run as administrator"
    echo.
)

REM Find Chrome
set CHROME_PATH=
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    set "CHROME_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe"
    echo [OK] Found Chrome
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    set "CHROME_PATH=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    echo [OK] Found Chrome
) else (
    echo [ERROR] Chrome not found!
    echo Please install Chrome from: https://www.google.com/chrome/
    pause
    exit /b 1
)

echo.
echo [1] Closing any existing Chrome instances...
taskkill /f /im chrome.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo.
echo [2] Setting up port forwarding (0.0.0.0:9222 -^> [::1]:9222)...
REM 删除旧的 v4tov4 规则
netsh interface portproxy delete v4tov4 listenport=9222 listenaddress=0.0.0.0 >nul 2>&1
REM 删除旧的 v4tov6 规则
netsh interface portproxy delete v4tov6 listenport=9222 listenaddress=0.0.0.0 >nul 2>&1
REM 添加新的 v4tov6 规则 (转发到 Chrome 的 IPv6 地址)
netsh interface portproxy add v4tov6 listenport=9222 listenaddress=0.0.0.0 connectport=9222 connectaddress=::1

echo.
echo [3] Adding firewall rule for port 9222...
netsh advfirewall firewall delete rule name="Chrome Remote Debug 9222" >nul 2>&1
netsh advfirewall firewall add rule name="Chrome Remote Debug 9222" dir=in action=allow protocol=TCP localport=9222 >nul 2>&1

echo.
echo [4] Starting Chrome with remote debugging on port 9222...
REM Chrome 144+ 默认使用 IPv6 [::1]:9222，我们通过 v4tov6 端口转发来解决
start "" "%CHROME_PATH%" --remote-debugging-port=9222 --user-data-dir="%USERPROFILE%\ChromeDebugProfile"

timeout /t 3 /nobreak >nul

echo.
echo [5] Verifying port forwarding...
echo --- v4tov4 rules ---
netsh interface portproxy show v4tov4
echo --- v4tov6 rules ---
netsh interface portproxy show v4tov6

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo   Chrome: Running on [::1]:9222 (IPv6)
echo   Port Forward: 0.0.0.0:9222 -^> [::1]:9222 (v4tov6)
echo   WSL2 Access: http://172.26.80.1:9222
echo ========================================
echo.
echo Keep this window open while using Agent browser control.
echo Press any key to exit (Chrome will keep running)...
pause >nul
