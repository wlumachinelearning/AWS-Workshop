
const { chromium } = require("playwright");
const path = require("path");
const { execFileSync } = require("child_process");

(async () => {
  const browser = await chromium.launch({
    headless: true
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    deviceScaleFactor: 1,
    recordVideo: {
      dir: "./recordings",
      size: { width: 1280, height: 720 }
    }
  });

  const page = await context.newPage();

  await page.goto(
    "file://" + path.resolve("visualization.html")
  );

  // Start auto-play from the beginning.
  await page.locator("#auto").click();

  // Wait until the final scene has started.
  await page.waitForFunction(
    () => document.querySelector("#title")
      ?.textContent === "The whole pipeline",
    { timeout: 300000 }
  );

  // Let the final scene finish animating.
  await page.waitForTimeout(22000);

  const video = page.video();

  await context.close();

  const recording = await video.path();
  await browser.close();

  execFileSync("ffmpeg", [
    "-y",
    "-i", recording,
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    "-movflags", "+faststart",
    "visualization.mp4"
  ]);

  console.log("Exported visualization.mp4");
})();
