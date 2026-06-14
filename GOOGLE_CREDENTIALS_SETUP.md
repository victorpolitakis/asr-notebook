# Google Drive OAuth Setup

How to create `credentials.json` for a new Google account (desktop OAuth app).

---

## 1. Google Cloud Console — Create or reuse a project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Top bar → project picker → **New Project** (or pick an existing one)
3. Name it anything (e.g. `asr-notebook`)

---

## 2. Enable the Drive API

1. Left menu → **APIs & Services → Library**
2. Search **Google Drive API** → click it → **Enable**

---

## 3. Create OAuth credentials

1. Left menu → **APIs & Services → Credentials**
2. **+ CREATE CREDENTIALS → OAuth client ID**
3. If prompted to configure consent screen:
   - User type: **External**
   - App name: anything
   - Support email: your email
   - Scroll to bottom → **Save and Continue** through all steps (no scopes needed here)
   - Back on Credentials page, repeat step 2
4. Application type: **Desktop app**
5. Name: anything → **Create**
6. Click **Download JSON** on the confirmation dialog (or the download icon next to the credential later)
7. Rename the file to `credentials.json`

---

## 4. Add yourself as a test user (required while app is in Testing)

1. Left menu → **APIs & Services → OAuth consent screen**
2. Scroll to **Test users** → **+ ADD USERS**
3. Add the Google account email you want to use → **Save**

> Without this step, the OAuth flow will show "Access blocked: app has not completed verification" for accounts other than the project owner.

---

## 5. Place the file

Put `credentials.json` in one of these locations (checked in order):

```
asr-notebook/Pipeline/credentials.json   ← preferred (next to drive_bridge.py)
asr-notebook/credentials.json
```

**Delete the old `token.json`** (same folder as credentials.json) so the app re-authenticates with the new account. The file is auto-created on first run.

---

## 6. First run

Start the Streamlit app → sidebar → **Connect Drive**. A browser tab opens for the OAuth flow. Sign in with the new account, grant Drive access. `token.json` is saved — subsequent runs are silent.

---

## Switching accounts again later

Delete `token.json` and repeat step 6. No need to touch Google Cloud Console.
