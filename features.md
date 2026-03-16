# AI Study Buddy Feature Roadmap

## Top Features to Build Next (Ranked)

1. **Study Mode session experience (Pomodoro + live coach overlay)**
   - One screen with timer, focus status, gentle nudges, and sprint recap.
   - Example recap: "Focused 18/25 min, biggest drift at 12:40, next goal: +10% focus time."
   - Why it matters: turns focus tracking into a clear product moment students feel immediately.

2. **Configurable unfocused nudge timing (sound alert)**
   - Add a user setting to choose how many minutes of being unfocused triggers a sound nudge.
   - This replaces the current hardcoded threshold with a student-controlled value.
   - Suggested UX:
     - Setting label: `Nudge me after X minutes unfocused`
     - Range: 1-15 minutes
     - Default: current hardcoded value (for backward compatibility)
     - Optional toggles: sound on/off, nudge volume
   - Why it matters: students have different attention styles and should control reminder sensitivity.

3. **Auto-generated Study Recap after each session (shareable)**
   - One-click recap card with:
     - what was studied (chat + session title)
     - key concepts
     - 3 takeaways
     - 5 quick quiz questions
     - focus stats
   - Export as image or PDF.

4. **Turn any chat/notes into quiz + flashcards**
   - Generate:
     - quick quiz (MCQ + short answer)
     - flashcards (spaced repetition-ready)
     - exam-style questions with rubric
   - Why it matters: converts passive reading into active recall quickly.
   - Execution plan: see `feature-4-implementation-tracker.md`.

5. **Upload/import course material, then chat with citations**
   - "Drop your PDF/DOCX/Excel/slides/TXT/images, then ask questions grounded in your material."
   - Even a basic retrieval flow with citations makes answers more trustworthy and course-specific.

6. **Streaks + lightweight gamification tied to focus minutes**
   - Daily streaks, weekly goals, badges, and a simple level system.
   - Progress should reward consistent focused time, not just app opens.

7. **Adaptive AI Study Coach (personalized in real time)**
   - Combine chat behavior + focus signals to dynamically coach each student during the session.
   - Example interventions:
     - "You drift every ~12 minutes; switching to 10-minute sprints for this session."
     - "You missed 3 derivative questions in a row; quick 2-minute refresher before continuing."
     - "Your focus drops after long answers; switching to step-by-step checkpoints."
   - End each session with a personalized next-session strategy based on what actually worked.
   - Why it matters: this makes the app feel like a truly personal tutor, not just a timer plus chatbot.

8. **Live Exam Simulation Mode (timed + adaptive difficulty)**
   - Students run a realistic mock exam with countdown timer, section pacing, and no-distraction interface.
   - Questions adapt in difficulty based on performance and confidence.
   - End with a performance breakdown: weak topics, time-loss moments, and a targeted recovery plan.
   - Why it matters: turns study time into exam readiness with immediate, actionable feedback.

## Quick Wow Features (1-2 Days Each)

1. **Nudge sound picker**
   - Let students pick from 3-5 built-in nudge sounds (bell, chime, soft knock).
   - Pairs well with configurable nudge timing and makes reminders feel more personal.
   - Estimated effort: **0.5-1 day**

2. **Session goal chip**
   - Prompt at session start: "What is your goal for this session?"
   - Keep the goal pinned near the timer and ask "Completed?" at session end.
   - Estimated effort: **1 day**

3. **One-tap "I'm back" button**
   - Show a quick action when distraction is detected so students can instantly reset.
   - Track "comeback count" to reinforce recovery behavior, not just perfection.
   - Estimated effort: **1 day**

4. **Instant celebration moments**
   - Trigger subtle confetti/sound when students hit milestones (first 25 focused minutes, streak day 3, etc.).
   - Adds delight without changing core workflow.
   - Estimated effort: **0.5-1 day**

5. **Session opener micro-challenge**
   - Before timer starts, suggest a 30-second challenge: "Close distracting tabs" or "Set one concrete objective."
   - Builds focus ritual and makes the app feel coach-like.
   - Estimated effort: **1 day**

6. **Smart break suggestion**
   - After long focused blocks, show a tiny prompt like "2-minute eye break?"
   - Keeps energy up and positions the app as supportive rather than punitive.
   - Estimated effort: **1 day**

7. **Keyboard shortcuts overlay**
   - Add a small `?` help modal with shortcuts for start/stop timer, mute nudge, and focus chat input.
   - Feels power-user friendly with low implementation cost.
   - Estimated effort: **1 day**

8. **Session mood check-in**
   - Quick emoji/1-click mood at start and end ("stressed", "okay", "focused").
   - Show mood trend in recap to give students a personal insight loop.
   - Estimated effort: **1-2 days**
