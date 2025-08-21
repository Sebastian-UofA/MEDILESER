# Gmail Bot Configuration Template
# Copy this file to config.py and customize your settings

class BotConfig:
    # Keywords that will trigger an auto-reply
    TRIGGER_KEYWORDS = [
        'support',
        'help', 
        'inquiry',
        'question',
        'assistance',
        'urgent',
        'problem'
    ]
    
    # Specific email addresses to monitor (leave empty to monitor all)
    TRIGGER_SENDERS = [
        # 'important@company.com',
        # 'client@domain.com'
    ]
    
    # Subject prefix for replies
    REPLY_SUBJECT_PREFIX = "Re: "
    
    # Your custom auto-reply message
    AUTO_REPLY_MESSAGE = """
    Hello!
    
    Thank you for reaching out to me. This is an automated response to confirm that I have received your email.
    
    I will review your message and respond personally as soon as possible, typically within 24 hours.
    
    If this is urgent, please feel free to call me directly.
    
    Best regards,
    Sebastian Perez
    Automated Response System
    """
    
    # How often to check for new emails (in seconds)
    CHECK_INTERVAL = 300  # 5 minutes
    
    # Whether to mark emails as read after processing
    MARK_AS_READ = False
