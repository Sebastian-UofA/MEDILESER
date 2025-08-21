# Gmail Bot Setup Instructions

## Prerequisites

1. **Google Cloud Project**: You need to create a Google Cloud project and enable the Gmail API.
2. **OAuth2 Credentials**: Download the credentials file to authenticate with Gmail.

## Step-by-Step Setup

### 1. Enable Gmail API

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API:
   - Go to "APIs & Services" > "Library"
   - Search for "Gmail API"
   - Click on it and press "Enable"

### 2. Create OAuth2 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Choose "Desktop application"
4. Give it a name (e.g., "Gmail Bot")
5. Download the JSON file
6. Rename it to `credentials.json` and place it in this folder

### 3. Configure the Bot

Edit the `gm.py` file to customize:

- **Trigger Keywords**: Words that will trigger an auto-reply
- **Trigger Senders**: Specific email addresses to monitor (optional)
- **Auto-reply Message**: Your custom response message
- **Reply Subject Prefix**: Prefix for reply subjects

### 4. Run the Bot

```powershell
# Navigate to the project folder
cd "z:\PERSONAL SMART\13. SEBASTIAN PEREZ\Python\Gmail bot"

# Run the bot
"z:/PERSONAL SMART/13. SEBASTIAN PEREZ/Python/Gmail bot/.venv/Scripts/python.exe" gm.py
```

## Configuration Options

### Trigger Keywords
```python
bot.trigger_keywords = ['support', 'help', 'inquiry', 'question', 'assistance']
```

### Specific Senders (Optional)
```python
bot.trigger_senders = ['important@company.com', 'client@domain.com']
```

### Custom Auto-Reply Message
```python
bot.auto_reply_message = """
Your custom message here...
"""
```

## Running Modes

1. **Run Once**: Test the bot by processing current unread emails once
2. **Run Continuously**: Monitor inbox continuously (recommended for production)

## Security Notes

- The `credentials.json` file contains sensitive information - keep it secure
- The bot will create a `token.pickle` file to store authentication tokens
- Never share these files publicly

## Troubleshooting

- Make sure Gmail API is enabled in your Google Cloud project
- Ensure `credentials.json` is in the correct location
- Check that your Google account has the necessary permissions
- Review the logs for any error messages

## Features

- ✅ Monitors Gmail inbox for new emails
- ✅ Filters emails based on keywords and senders
- ✅ Sends automated replies
- ✅ Prevents reply loops (won't reply to emails starting with "Re:")
- ✅ Tracks processed emails to avoid duplicates
- ✅ Comprehensive logging
- ✅ Configurable check intervals
- ✅ Test mode for safe testing
