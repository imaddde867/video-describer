# Independent multimodal analysis — GLM-5.3 Flash

This comparison was produced in fresh OpenCode sessions using `opencode/glm-5.3-flash-free`, separately from the system pipeline. The completed independent GLM analyses cover **2 of the 9 videos** (the two reports below); the seven remaining sessions were launched with their respective clips attached directly and began analysis, but OpenCode stopped each with `Free usage limit reached` before a final answer. The seven incomplete attempts are recorded individually in the `GLM_*.md` status notes; they are not claimed as summaries. Consequently, a completed nine-video GLM-vs-pipeline comparison could not be produced in this run. The source pipeline's full nine-video results remain available in `RESULTS.md` and each per-video `description.md`.

The completed GLM work used fresh sessions and did not read the pipeline reports. The original MP4s exceeded the agent's attachment-size limit, so local 360×640 H.264/AAC copies (≤2.2 MB) were attached. They retain moving video and sound but reduce visual detail. For both completed reports the agent used sampled frames and ran faster-whisper on attached audio. An earlier attempt selected the wrong recent Downloads MP4 after failing to see its attachment; that run was stopped and excluded, then the exact requested clips were explicitly attached in clean runs.

> **Comparison caveats:** For clip 1, despite AAC being present in the attached file, GLM incorrectly said audio was not exported and did not produce a transcript; its voice/music observations are incomplete. For clip 2, GLM verified AAC, used 43 one-second frames, and generated a local Whisper transcript. Timestamps and small print are approximate because of re-encoding and sparse sampling. The other seven sessions hit the provider's free-usage limit, not a video-decoding or pipeline failure; no claims about their GLM visual/audio analysis are made.

| Video | System pipeline (Qwen VL + Qwen Audio; Whisper where available) | Fresh GLM-5.3 Flash |
|---|---|---|
| `20261025_164350000_iOS` | Atticus Mac voice-assistant promotion; explains voice-driven app/window/file actions, draft email and learned focus shortcut. Pipeline combines visual sampled frames with a separate audio-model analysis. | Same Atticus project and on-screen examples; usefully flags the displayed UI as marketing/mockup rather than verified live behavior. GLM did not transcribe the AAC audio, so its spoken-content coverage is weaker here. |
| `20261025_164610000_iOS` | Mac AI buddy/robot build with 3D-printed head, ESP32-S3 camera, SG90 servos, Cursor agents and a face-control web UI. | Independently reaches the same main project/hardware story and adds a local Whisper transcript; distinguishes narration claims (tracking face/objects) from demonstrated prototype behavior. |
| Other 7 videos | Complete pipeline reports below. | No completed GLM result: provider free-usage cap stopped those sessions. |

## Clip 1 — `20261025_164350000_iOS.MP4`

### Summary

A ~49-second vertical Instagram explainer/ad for **Atticus** (`atticus-mac` on GitHub), a Mac voice assistant pitched as “Jarvis for your Mac.” It opens with a face-to-camera segment in a dark room and cuts to animated UI mockups: a blue mascot, kinetic captions, a Mac listening bar, an email reply draft, learned shortcuts, and a Claude-plan requirement. It closes with a comment “ATTICUS” call to action and a simulated DM with an install guide. The handle shown is **@antoniorevenue**. The “README EXAMPLE” / “EXAMPLE” labels indicate stylized mockups rather than proof of a real product session. The pipeline independently observed the same product and captions on this clip, but described the animated demo as app interaction; the mockup caveat is a useful distinction from GLM.

The pipeline's separate audio lane reports a male presenter explaining voice-driven app/window/file actions, email replies, learned “focus mode,” Claude Pro/Max or local-model requirements, and macOS 26. Its audio interpretation is available in the per-video report; those are stronger speech claims than GLM's limited audio check.

### Approximate timeline (GLM; ~5-second frame spacing)

| Time | Visuals and content |
|---|---|
| 0:00–0:03 | “Jarvis for your Mac”; repo card `atticus-mac`, “OPEN SOURCE · MIT,” “Hey Jarvis”; presenter at a microphone. |
| 0:03–0:08 | “like a person”; Atticus card “Your whole Mac, one sentence away”; listening bar: “Open Spotify. Terminal on the left.” |
| 0:08–0:12 | Kinetic text introduces “it needs first one thing” (teaser for the requirement). |
| 0:12–0:17 | “and tell it what you need”; Finder-style window and shortcut chips (Control, Option, Space); “README EXAMPLE.” |
| 0:17–0:22 | “and that's your workday”; 9 AM–5 PM slider card. |
| 0:22–0:27 | “And it can”; Mail mockup from Alex Kim, “Reading the window / Looking at your screen,” draft reply “Hey, reply to this,” tagged “DRAFT · NEVER SENDS.” |
| 0:32–0:37 | “it learns” / “add block”; learned shortcut “focus mode” → close Slack + open to-do list; “saved,” “NO-AI · INSTANT”; Reminders items. |
| 0:37–0:42 | “The catch is your Claude plan”; card says app is free and runs on Claude Pro/Max or a local model such as Ollama/LM Studio. |
| 0:42–0:45 | “Want the setup?”; comment box mockup. |
| 0:45–0:49 | Install-guide DM mockup; CTA to comment “ATTICUS” and follow. |

The gap between 0:27 and 0:32 reflects sparse visual sampling; intermediate events were not established by this independent run.

### Technical details visible in the video

- **App:** Atticus for macOS; GitHub project/repo label `atticus-mac`; depicted as open source under MIT.
- **Interaction:** hold Control + Option + Space and speak; a local listening/transcription bar is depicted.
- **Automation examples:** open Spotify and position Terminal; read the current window; draft (not send) an email; learn “focus mode” to close Slack and open a to-do list.
- **Model backends disclosed:** Claude Pro/Max plan or a local model through Ollama / LM Studio; the app itself is labeled free.
- **Permissions shown in the install-guide mockup:** Accessibility and Screen Recording.
- **Visual caveat:** These are marketing UI mockups. GLM did not validate that the depicted app behavior works in an actual macOS environment.

### Independent audio observations and limitations

GLM noted continuous voiceover and an energetic creator-style delivery, with measured RMS roughly −13 to −18 dB and no ≥1.5 s silence at its −32 dB threshold. It did **not** produce a transcript; its report said no audio was exported, although ffprobe confirms AAC exists in both source and re-encoded attachment. Treat the speech/music observations as incomplete and not equivalent to the pipeline's Qwen2-Audio analysis.

### Uncertainties

- Timestamps are approximate from sparse visual sampling (roughly ±2 seconds); short transitions may be missed.
- Fine print / exact wording in the final install-guide mockup is partly illegible at the reduced resolution.
- Exact spoken wording and any subtle music bed were not independently verified by ASR or full audio-language understanding.
- The model saw a re-encoded 360×640 copy, not the original 720×1280 MP4.

---

## Independent session outcome for the other seven videos

Each clip was separately attached to a fresh `opencode/glm-5.3-flash-free` session, with instructions to analyze only that clip and not inspect code or prior reports. The sessions started inspecting video/audio but ended before returning a final report because the provider responded **“Free usage limit reached. Connect a provider or subscribe to OpenCode Go to keep going, or switch models.”** No final GLM summaries were produced for these files:

- `20261025_164620000_iOS.MP4`
- `20261025_164935000_iOS.MP4`
- `20261025_165244000_iOS.MP4`
- `20261025_165416000_iOS.MP4`
- `20261025_165443000_iOS.MP4`
- `20261025_165745000_iOS.MP4`
- `20261025_173551000_iOS.MP4`

The matching `GLM_<video>.md` files document this limitation and link to the system report. This avoids mistaking partial scratch notes for an independent result. Re-run these seven GLM jobs after OpenCode provider usage is available to complete the requested like-for-like comparison.

---

## Clip 2 — `20261025_164610000_iOS.MP4`

### Summary

A 42.8-second “Day 2” progress reel by **Aik Han** (handle `@AYKHANIUM`), showing **Mac**, a white 3D-printed AI robot / companion. The creator adds a physical body to an earlier talking-only version, with screen-face expressions, a camera, face/object tracking claims, and two servos for movement. Cursor is shown coordinating parallel AI coding agents: one for movement/face-tracking logic, another for a realtime web face-preview/control site called **face deck**. The reel ends with a comment “MINIBOT” CTA. The video shows a maker-demo prototype rather than evidence of production readiness.

### Approximate timeline (GLM; 43 sampled frames at 1 fps)

| Time | Observed content |
|---|---|
| 0:00–0:03 | “Day 2 of building my AI buddy with Cursor”; on-screen “AI BUDDY.” |
| 0:03–0:07 | Creator introduces himself and the robot as Aik Han and Mac; shows the white printed head with a small pixel-expression screen and camera on top. |
| 0:07–0:12 | The robot is held/rotated; narration says Mac could previously only talk, so a physical body was added; narration claims it can take in surroundings, track the creator’s face and held objects. |
| 0:12–0:17 | Close-ups of hardware; narration identifies two servo motors and an ESP32-S3 camera. |
| 0:17–0:22 | Wiring and printed body parts; maker notes there were many wires. |
| 0:22–0:25 | “After building the hardware it was time to bring Mac to life.” |
| 0:25–0:32 | Cursor interface with several agents; one handles movement/face tracking, another the web interface. |
| 0:32–0:36 | “face deck” site: emotion tiles, selected “looking” state, green robot-face preview visibly matching Mac’s display. |
| 0:36–0:43 | CTA: comment “MINIBOT” for project link; `@AYKHANIUM` end card. |

### Technical details

- **Robot:** white 3D-printed Macintosh-like head, black-bezel low-resolution screen face, top camera module, neck / two-axis stand and printed base.
- **Motion hardware:** narration names **two servo motors**; the creator also holds up blue SG90 micro-servos. Pan/tilt is a reasonable inference from the visible two-axis mechanism, not a verified control mapping.
- **Vision hardware:** creator explicitly says **ESP32-S3 camera**; a camera module is visible on top of the head. Typed Cursor prompt mentions ESP32 + OLED, but display type itself is not verifiable from pixels.
- **Software workflow:** Cursor, multiple concurrent agents (movement + face tracking; web interface), site called “face deck.” The UI includes a live preview and about 15 emotion tiles (labels are best-effort due to low resolution).
- **Outcome shown:** robot expression and web preview appear synchronized; a small picture-in-picture shows a “human” bounding box. Narrated object tracking is a claim; the clip does not establish robustness or show a test matrix.
- **Not visible/verified:** exact servo wiring/controller board, power source, transport protocol, frame rate/latency, model used for tracking, or whether the software/site is published.

### Spoken claims (Whisper small, locally transcribed by the independent agent)

- 0:00–0:02: “This is day 2 of building my AI body with Cursor.” **ASR heard “body”; burned-in title says “BUDDY.”**
- 0:02–0:07: “My name is Aik Han and this is Mac. Mac was only able to talk to me so I gave him a new body.”
- 0:07–0:12: “Now we can absorb everything around him, track my face and whatever I'm holding.” (Narrated capability, not validated by this demo.)
- 0:12–0:17: “For body movements I use 2 servo motors and the camera from ESP32 S3.”
- 0:17–0:22: assembly took time because there were “so many wires.”
- 0:22–0:36: hardware complete; Cursor launched multiple agents; one handled movement/face-tracking logic while another built the site for realtime tracking/preview of Mac's faces.
- 0:36–0:39: “Comment mini-bot below and I will send you the link to the project.” On-screen CTA spells **MINIBOT**.

### Uncertainties / quality notes

- The attached re-encode is 360×640, H.264 30 fps, mono AAC 24 kHz; this is derived metadata, not original camera quality.
- Frame sampling at about 1 fps can miss quick assembly or interaction details; text in the Cursor prompt, agent labels, face tiles, and sliders is approximate.
- Speech ASR is helpful for exact part names, but “AI body/buddy” demonstrates an ASR ambiguity; visual captions were used to resolve it.
- WebSocket/networking and pan/tilt are plausible implementation inferences, not observed facts. Do not treat spoken face/object tracking claims as validated performance.
