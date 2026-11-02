# Coal on the phone (Tecno Pova 6 Neo, no root)

One Coal: the laptop is where it is taught; the phone gets a copy every time you plug it in.
These are the steps that worked on 2026-11-01. Type each line on its own and press Enter.

## Once, on the phone
1. Install **Termux** from F-Droid (f-droid.org). The Play Store version is old and broken.
2. In Termux:
   ```
   termux-change-repo
   ```
   Pick **Mirror group**, then **All mirrors** (an old mirror gives "cannot be downloaded").
3. Then:
   ```
   apt update -y
   apt install -y tur-repo
   apt update -y
   apt install -y python python-numpy python-torch
   termux-setup-storage
   ```
   `tur-repo` is the extra package source PyTorch lives in. `termux-setup-storage` asks to see
   your files: allow it.
4. Check PyTorch works:
   ```
   python -c "import torch; print(torch.__version__)"
   ```
   It should print a version number (2.15.0 on 11-01).

**Never use `pip` for numpy or torch in Termux.** `pip uninstall` deletes the files apt put there,
and `pip install torch` has no phone build. If torch goes missing ("No module named torch") while
apt says it is installed, put the files back with:
```
apt install --reinstall -y python-torch python-numpy
```

## Every time
1. Plug the phone into the laptop and pick **File transfer** on the phone's USB popup.
   The laptop copies the newest Coal to the phone's storage, in a folder called **Coal**, by itself
   (only if Coal changed). It may land on the SD card, e.g. `/storage/69F7-EBAB/Coal`.
2. In Termux, go to that folder and start Coal:
   ```
   cd /storage/69F7-EBAB/Coal
   python -m household.chat
   ```
   (or `cd ~/storage/shared/Coal` if it landed in internal storage). Type to Coal; `quit` to stop.

## Good to know
- Copies go laptop -> phone only. Anything Coal learns on the phone (your name, new words) is
  replaced by the laptop's Coal at the next plug-in. Teach it on the laptop.
- No microphone or screenshots on the phone yet: chat only.
- Log of every copy: `%APPDATA%\CoalSync.log` on the laptop.
