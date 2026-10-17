// The first-start progress window: shows what the component install (uv sync) is doing, read from its log, until
// the launcher closes it. Plain JavaScript for Automation, so it runs on any Mac without anything installed:
//
//   osascript -l JavaScript progress.js <install log>
ObjC.import("Cocoa");

const SIZE_UNITS = { KiB: 1024, MiB: 1024 ** 2, GiB: 1024 ** 3 };

function megabytes(bytes) {
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

// What uv has written so far, as { text, fraction } (fraction null: no measure yet).
function readStatus(logPath) {
  const contents = $.NSString.stringWithContentsOfFileEncodingError(logPath, $.NSUTF8StringEncoding, null);
  const lines = contents.isNil() ? [] : contents.js.split("\n");
  const sizes = {};
  let total = 0;
  let done = 0;
  let text = "Getting ready…";
  let fraction = null;
  for (const line of lines) {
    let m;
    if (/Downloading cpython/i.test(line)) {
      text = "Downloading Python…";
    } else if ((m = line.match(/^\s*Downloading (\S+) \(([\d.]+)\s*(KiB|MiB|GiB)\)/))) {
      sizes[m[1]] = parseFloat(m[2]) * SIZE_UNITS[m[3]];
      total += sizes[m[1]];
    } else if ((m = line.match(/^\s*Downloaded (\S+)/))) {
      done += sizes[m[1]] || 0;
    } else if ((m = line.match(/^\s*Building (\S+)/))) {
      text = `Building ${m[1]}…`;
    } else if (/^\s*(Prepared|Installed) /.test(line)) {
      text = "Installing packages…";
      fraction = null;
      total = 0;  // downloads are over
    }
    if (total > 0 && done < total) {
      text = done > 0 ? `Downloading packages: ${megabytes(done)} of ${megabytes(total)}`
                      : `Downloading packages (${megabytes(total)})…`;
      fraction = done > 0 ? Math.min(done / total, 1) : null;
    }
  }
  return { text, fraction };
}

function label(text, frame, font) {
  const field = $.NSTextField.wrappingLabelWithString(text);
  field.setFrame(frame);
  if (font) field.font = font;
  return field;
}

function run(argv) {
  const logPath = argv[0];
  const app = $.NSApplication.sharedApplication;
  app.setActivationPolicy($.NSApplicationActivationPolicyAccessory);  // no Dock icon, like the editor

  const win = $.NSWindow.alloc.initWithContentRectStyleMaskBackingDefer(
    $.NSMakeRect(0, 0, 460, 170), $.NSWindowStyleMaskTitled, $.NSBackingStoreBuffered, false);
  win.title = "Qwen Image Editor";
  win.level = $.NSFloatingWindowLevel;  // no Dock icon to find it by, so it stays in front
  const view = win.contentView;
  view.addSubview(label("Installing components", $.NSMakeRect(20, 125, 420, 22), $.NSFont.boldSystemFontOfSize(14)));
  const status = label("Getting ready…", $.NSMakeRect(20, 100, 420, 20));
  view.addSubview(status);
  const bar = $.NSProgressIndicator.alloc.initWithFrame($.NSMakeRect(20, 72, 420, 20));
  bar.minValue = 0;
  bar.maxValue = 1;
  bar.indeterminate = true;
  bar.startAnimation(null);
  view.addSubview(bar);
  const footer = label("", $.NSMakeRect(20, 14, 420, 48), $.NSFont.systemFontOfSize(11));
  footer.textColor = $.NSColor.secondaryLabelColor;
  view.addSubview(footer);

  win.center;
  win.makeKeyAndOrderFront(null);
  app.activateIgnoringOtherApps(true);

  const started = Date.now();
  for (;;) {
    const { text, fraction } = readStatus(logPath);
    status.stringValue = text;
    if (fraction === null) {
      bar.indeterminate = true;
      bar.startAnimation(null);
    } else {
      bar.indeterminate = false;
      bar.doubleValue = fraction;
    }
    const seconds = Math.floor((Date.now() - started) / 1000);
    const elapsed = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    footer.stringValue = `${elapsed} so far. This happens once, on the first start, and takes a few minutes ` +
      "depending on your internet connection. The editor opens in your browser when it's done.";
    win.display;
    $.NSRunLoop.currentRunLoop.runUntilDate($.NSDate.dateWithTimeIntervalSinceNow(0.5));
  }
}
