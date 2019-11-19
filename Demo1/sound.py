# from pydub import AudioSegment
# from pydub.playback import play

# sound = AudioSegment.from_wav('beep-06.wav')
# play(sound)

# print("\a")

import subprocess
subprocess.call(["afplay", "beep-06.wav"])
