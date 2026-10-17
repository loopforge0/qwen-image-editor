; Inno Setup script for the Windows installer. Build with installer\windows\build.ps1, which stages the files first.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "Qwen Image Editor"
#define Stage "build\stage"

[Setup]
AppId={{8C3F6A52-4E1B-4D7A-9B2E-6F1D0A7C3E91}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Qwen Image Editor contributors
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user install: no administrator prompt.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=QwenImageEditor-Setup-{#AppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\icon.ico
UninstallDisplayName={#AppName}
InfoBeforeFile=info.txt
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
; One setup at a time: a second one says setup is already running instead of installing over the first.
SetupMutex=QwenImageEditorSetup
; Every installed file gets the same fixed date, so the installer doesn't record when (or how long) it was built.
TouchDate=2026-01-01
TouchTime=00:00

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
; Only offered for the chat apps that are installed. Connecting adds the editor (its MCP server) to their settings.
Name: claude; Description: "Claude Desktop"; GroupDescription: "Use the editor from chat apps:"; Check: ClaudeFound
Name: goose; Description: "Goose"; GroupDescription: "Use the editor from chat apps:"; Check: GooseFound
Name: chatgpt; Description: "ChatGPT"; GroupDescription: "Use the editor from chat apps:"; Check: ChatGPTFound

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion touch

[Icons]
; uvw.exe is uv without a console window; qie-tray is the tray app (it opens the editor in the browser).
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\uvw.exe"; Parameters: "run --frozen qie-tray"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\uvw.exe"; Parameters: "run --frozen qie-tray"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
; The component install and connecting the chat apps happen in InstallComponents (below), which can retry.
; Started through Explorer and the Start menu shortcut, not as a child of setup: setup turns on Windows'
; RedirectionGuard, which its child processes inherit, and that refuses the symbolic links in the Hugging Face cache
; the model loads from (WinError 448, "untrusted mount point").
Filename: "{win}\explorer.exe"; Parameters: """{autoprograms}\{#AppName}.lnk"""; Description: "Start {#AppName}"; Flags: postinstall nowait skipifsilent; Check: ComponentsInstalled

[UninstallRun]
; Take the editor out of the chat apps' settings, where it runs this install (not where it runs another copy).
Filename: "{app}\uv.exe"; Parameters: "run --frozen qie disconnect --this-install"; WorkingDir: "{app}"; Flags: runhidden waituntilterminated; RunOnceId: "DisconnectChatApps"

[InstallDelete]
; Left over from 0.0.1, which started the editor through a console window.
Type: files; Name: "{app}\Qwen Image Editor.cmd"

[UninstallDelete]
; The Python environment created on first start.
Type: filesandordirs; Name: "{app}\.venv"
Type: dirifempty; Name: "{app}"

[Code]
// The chat apps the editor can be connected to, as far as can be told without starting them.
function ClaudeFound(): Boolean;
var
  FindRec: TFindRec;
begin
  Result := DirExists(ExpandConstant('{userappdata}\Claude')) or DirExists(ExpandConstant('{localappdata}\AnthropicClaude'));
  if not Result and FindFirst(ExpandConstant('{localappdata}\Packages\Claude_*'), FindRec) then
  begin
    Result := True;
    FindClose(FindRec);
  end;
end;

function GooseFound(): Boolean;
begin
  Result := DirExists(ExpandConstant('{userappdata}\Block\goose')) or DirExists(ExpandConstant('{localappdata}\Programs\Goose'));
end;

// The ChatGPT desktop app keeps its MCP servers in the Codex settings (.codex in the user's folder); its Store
// package is named OpenAI.Codex.
function ChatGPTFound(): Boolean;
var
  FindRec: TFindRec;
begin
  Result := DirExists(ExpandConstant('{%USERPROFILE}\.codex'));
  if not Result and FindFirst(ExpandConstant('{localappdata}\Packages\OpenAI.Codex_*'), FindRec) then
  begin
    Result := True;
    FindClose(FindRec);
  end;
end;

// A running editor holds its files open (uvw.exe, the Python processes), and having no window, it can't be closed
// by Windows' Restart Manager. So stop it first: ask it to quit like its Quit button does, then end whatever is
// still running from the install folder, with its child processes (the engine). Not the uninstaller, which runs
// from there too: ending it would end its second stage (a child process) and the uninstall with it.
procedure StopRunningEditor();
var
  ResultCode: Integer;
begin
  Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    '-NoProfile -NonInteractive -Command "try { Invoke-RestMethod -Method Post -TimeoutSec 3 http://127.0.0.1:7861/api/quit | Out-Null; Start-Sleep -Seconds 3 } catch {}; ' +
    'Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith(''' + ExpandConstant('{app}') + ''', [StringComparison]::OrdinalIgnoreCase) -and $_.ProcessName -notlike ''unins*'' } | ' +
    'ForEach-Object { taskkill /F /T /PID $_.Id | Out-Null }"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

var
  Installed: Boolean;

function ComponentsInstalled(): Boolean;
begin
  Result := Installed;
end;

// The one-time component install (Python and packages, about 2 GB to download), in a console window that shows its progress.
// Setup can't tell a failed install from a good one by itself, and if it went on, the first start would redo the
// install with no window at all, looking like nothing happens. So: on an error the console stays open to show it,
// then setup offers to try again.
procedure InstallComponents();
var
  ResultCode: Integer;
  OnError: String;
begin
  WizardForm.StatusLabel.Caption := 'Installing components (about 2 GB to download; a few minutes)...';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  OnError := 'echo. & echo Installing components failed. & pause & exit 1';
  if WizardSilent then
    OnError := 'exit 1';  // nobody to press a key
  repeat
    Installed := Exec(ExpandConstant('{cmd}'), '/c uv.exe sync --frozen || (' + OnError + ')',
      ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  until Installed or (SuppressibleMsgBox('Installing components failed (the window that just closed shows why). ' +
      'Check the internet connection, then try again.' + #13#10#13#10 +
      'If you cancel, run this setup again later to finish: the editor can''t start without its components.',
      mbError, MB_RETRYCANCEL, IDCANCEL) <> IDRETRY);
  WizardForm.ProgressGauge.Style := npbstNormal;
  if not Installed then
    Exit;

  // Connecting a chat app adds the editor (its MCP server) to that app's settings.
  if WizardIsTaskSelected('claude') then
  begin
    WizardForm.StatusLabel.Caption := 'Connecting to Claude Desktop...';
    Exec(ExpandConstant('{app}\uv.exe'), 'run --frozen qie connect claude', ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
  if WizardIsTaskSelected('goose') then
  begin
    WizardForm.StatusLabel.Caption := 'Connecting to Goose...';
    Exec(ExpandConstant('{app}\uv.exe'), 'run --frozen qie connect goose', ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
  if WizardIsTaskSelected('chatgpt') then
  begin
    WizardForm.StatusLabel.Caption := 'Connecting to ChatGPT...';
    Exec(ExpandConstant('{app}\uv.exe'), 'run --frozen qie connect chatgpt', ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    InstallComponents();
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  StopRunningEditor();
  Result := '';
end;

function InitializeUninstall(): Boolean;
begin
  StopRunningEditor();
  Result := True;
end;

// Models and history live outside the program folder. Ask before deleting them; models used from a ComfyUI folder
// are never touched.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\qwen-image-editor');
    if DirExists(DataDir) then
      // SuppressibleMsgBox: a silent uninstall (/SUPPRESSMSGBOXES) gets the default answer, No (keep the data).
      if SuppressibleMsgBox('Also delete the downloaded models (up to 17 GB) and your image history?' + #13#10#13#10 + DataDir,
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
