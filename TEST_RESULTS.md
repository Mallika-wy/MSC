# Executed Validation

Validation uses the existing OMG environment at
`/home1/gmh/wangyang/motion/OMG/.venv/bin/python`; no new environment or model
download was required.

| Check | Result |
| --- | --- |
| Fast unit tests | **27 passed**, 5 slow/integration tests deselected |
| 5x30 smoke training | Two CPU optimization steps completed; loss `1.2812 -> 1.2557` |
| 5x30 checkpoint sampling | Generated finite canonical motion `[1,30,125]` |
| Test-time RTC | Synthetic model sampling completed with PiGDM/VJP guidance |
| Train-time RTC | Frozen-prefix loss and sampling behavior covered by unit tests |

The current motion contract is four historical frames plus the current frame
(`[B,5,125]`) and 30 future frames (`[B,30,125]`). Audio uses one 35D feature
row for each predicted motion frame.

The local `/home1/gmh/wangyang/motion/OMG/data/OMG-Data` directory currently
contains a text-only LeRobot v3.0 release. Its metadata lacks
`omg.audio.feature` and the G1 motion fields required by MSC's official-data
adapter, so real 125D/audio numerical-parity tests were not run against it.
They remain opt-in for a matching OMG release and local T5 assets.

These checks cover tensor contracts and execution only. Motion quality requires
training and evaluation on a matching full OMG motion release.
