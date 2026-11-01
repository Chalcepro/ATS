# Coal on the phone (Tecno Pova 6 Neo, no root)

One Coal: the laptop is where it is taught; the phone gets a copy every time you plug it in.

## Once, on the phone
1. Install **Termux** from F-Droid (f-droid.org). The Play Store version is old and broken.
2. Open Termux and run, one line at a time:
   ```
   pkg update
   pkg install python python-torch python-numpy
   termux-setup-storage
   ```
   (`termux-setup-storage` asks to see your files: allow it.)

## Every time
1. Plug the phone into the laptop and pick **File transfer** on the phone's USB popup.
   The laptop copies the newest Coal to *Internal storage/Coal* by itself (only if it changed).
2. In Termux:
   ```
   cd ~/storage/shared/Coal
   python -m household.chat
   ```
   Type to Coal; `quit` to stop.

## Good to know
- Copies go laptop -> phone only. Anything Coal learns on the phone (your name, new words) is
  replaced by the laptop's Coal at the next plug-in. Teach it on the laptop.
- No microphone or screenshots on the phone yet: chat only.
- Log of every copy: `%APPDATA%\CoalSync.log` on the laptop.
