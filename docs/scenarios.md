# 🎬 Interaction scenarios

Run each scenario separately. Arrange the windows before pressing **Space** in the preview. **Q / Esc** closes the preview. The spoken examples below are rehearsal suggestions, not exact phrases required by ASR.

## 1 · 👋 Introduction and object awareness

**One person**

```bash
.venv/bin/python main.py --people=1 --scenario=introduction --mic=builtin --object-model=lite0
```

1. Enter/settle in view. Furhat follows and greets you.
2. When asked your name: “My name is Alice.”
3. Answer the day question naturally while holding a clearly visible bottle/cup.
4. Furhat acknowledges the answer and observed object; answer its follow-up question.

**Two people**

```bash
.venv/bin/python main.py --people=2 --scenario=introduction --mic=builtin --object-model=lite0
```

1. Alice begins alone, introduces herself and answers the day question.
2. Bob enters **while Alice is answering**, holding a bottle/cup.
3. Furhat finishes Alice's turn, welcomes Bob and asks his name.
4. Bob introduces himself. Furhat acknowledges the detected object he is holding and asks about it.
5. Bob answers, for example: “It's water. I brought it for the game.”

Keep the object visible. Object labels are actual predictions and can be wrong; try SSD/Lite0/Lite2 if one detector struggles with the prop.

## 2 · ✋ Rock–paper–scissors

**One person**

```bash
.venv/bin/python main.py --people=1 --scenario=game --mic=builtin --name=Alice
```

1. Answer the hobby question: “I like playing games.”
2. Agree to rock–paper–scissors.
3. Shake your head when asked if you know the rules.
4. Follow the explanation; show a clear sign after each countdown.
5. Play three rounds. Furhat announces results and celebrates wins with a wink before its closing thanks.

The one-person scene uses **precommitted robot moves: paper, scissors, rock**. To rehearse a 2–1 Furhat win for reflection, show scissors, paper, scissors. The detected signs are scored normally; a different sequence can change the winner. This is a rehearsed opponent, not a randomized one.

**Two people**

```bash
.venv/bin/python main.py --people=2 --scenario=game --mic=builtin --names Alice Bob
```

1. Sit in preview order: Alice left, Bob right. Each answers the hobby question.
2. Furhat explicitly suggests the game and acts as judge.
3. Alice says she knows the rules; Bob shakes his head. Furhat acknowledges the answers and explains.
4. After each countdown, show **one hand each**: closed fist = rock, open palm = paper, victory sign = scissors.
5. Hold the signs steady, then lower hands between rounds. Three detected rounds determine the result.

Keep hands reasonably close to their owner's body. Avoid crossing arms into the other participant's area. The referee scores observed signs; there is no guaranteed winner.

## 3 · 💬 Reflection, acknowledgment and interruption

**One person**

```bash
.venv/bin/python main.py --people=1 --scenario=reflection --mic=builtin --name=Alice
```

**Two people** — replace `Bob` with the actual winner:

```bash
.venv/bin/python main.py --people=2 --scenario=reflection --mic=builtin --names Alice Bob --winner Bob --profile
```

1. Face normally before Space to establish the settled posture reference.
2. In two-person mode, the winner answers first: “It was fun; I enjoyed it.”
3. The other participant answers while looking down: “It was quite bad. I think it could have been better.”
4. Furhat nods while listening, acknowledges the observed downward cue and reassures the participant.
5. Answer: “Okay.”
6. **During** Furhat's subsequent encouragement, interrupt: “Wait, I feel so stupid.” The UI shows when interruption monitoring is enabled.
7. If demonstrating the near-face hand cue, **mime movement without making contact**. Furhat responds to the spoken/visual context with reassurance and a reminder against self-harm.
8. Respond: “Okay, you're right.” The scenario closes.

The demo recognizes observable motion and spoken context; it does not verify harm or infer a diagnosis. Only show a safe staged gesture.

## ✅ Recording checklist

- Robot voice/lip sync tested; Ollama running.
- Correct microphone selected and actual device name confirmed in the terminal.
- Camera, perception and robot windows arranged before Space.
- Two-person names match **left-to-right preview order**; reflection winner matches the game.
- Speak one at a time. These scenarios do not separate simultaneous speech.
- Close each scenario before starting the next. Two-person runs save no run files; single-person local outputs are ignored by Git.

[← Back to the README](../README.md)
