import os
import base64
import time
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GmailBot:
    def __init__(self):
        self.SCOPES = [
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/gmail.send'
        ]
        self.service = None
        self.processed_emails = set()  # To avoid processing the same email twice in current session
        
        # Configuration - Customize these settings
        self.trigger_keywords = ['BOLETA - MEDILESER S.A.C', 'CONTRATO DE TRABAJO - MEDILESER S.A.C']  # Keywords to trigger auto-reply
        self.trigger_senders = ['docuementoslaborales@medileser.com.pe']  # Specific email addresses to monitor (leave empty to monitor all)
        self.reply_subject_prefix = "Re: "
        self.auto_reply_message = """Recibi conforme"""
        
    def authenticate(self):
        """Authenticate and build the Gmail service."""
        creds = None
        
        # Check if token.pickle exists (saved credentials)
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no valid credentials, request authorization
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail service authenticated successfully")
    
    def get_unread_emails(self):
        """Get unread emails from inbox."""
        try:
            results = self.service.users().messages().list(
                userId='me', 
                q='is:unread in:inbox'
            ).execute()
            
            messages = results.get('messages', [])
            logger.info(f"Found {len(messages)} unread emails")
            return messages
            
        except Exception as error:
            logger.error(f"Error getting unread emails: {error}")
            return []
    
    def get_email_details(self, message_id):
        """Get detailed information about an email."""
        try:
            message = self.service.users().messages().get(
                userId='me', 
                id=message_id,
                format='full'
            ).execute()
            
            payload = message['payload']
            headers = payload.get('headers', [])
            
            # Extract email details
            email_details = {
                'id': message_id,
                'thread_id': message['threadId'],
                'subject': '',
                'sender': '',
                'body': '',
                'snippet': message.get('snippet', '')
            }
            
            # Parse headers
            for header in headers:
                name = header['name'].lower()
                if name == 'subject':
                    email_details['subject'] = header['value']
                elif name == 'from':
                    email_details['sender'] = header['value']
            
            # Extract body
            email_details['body'] = self.extract_body(payload)
            
            return email_details
            
        except Exception as error:
            logger.error(f"Error getting email details: {error}")
            return None
    
    def extract_body(self, payload):
        """Extract email body from payload."""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
                    break
        else:
            if payload['mimeType'] == 'text/plain':
                data = payload['body']['data']
                body = base64.urlsafe_b64decode(data).decode('utf-8')
        
        return body
    
    def should_auto_reply(self, email_details):
        """Determine if an email should trigger an auto-reply."""
        # Check if we already processed this email
        if email_details['id'] in self.processed_emails:
            return False
        
        # Check for specific senders if configured
        if self.trigger_senders:
            sender_match = any(sender.lower() in email_details['sender'].lower() 
                             for sender in self.trigger_senders)
            if not sender_match:
                return False
        
        # Check for trigger keywords in subject or body
        subject_lower = email_details['subject'].lower()
        body_lower = email_details['body'].lower()
        snippet_lower = email_details['snippet'].lower()
        
        keyword_found = any(
            keyword.lower() in subject_lower or 
            keyword.lower() in body_lower or 
            keyword.lower() in snippet_lower
            for keyword in self.trigger_keywords
        )
        
        # Don't reply to emails that are already replies (to avoid loops)
        if email_details['subject'].lower().startswith('re:'):
            return False
        
        return keyword_found
    
    def send_reply(self, original_email):
        """Send an automated reply to an email."""
        try:
            # Create reply message
            reply = MIMEMultipart()
            reply['to'] = original_email['sender']
            reply['subject'] = self.reply_subject_prefix + original_email['subject']
            
            # Add body
            reply.attach(MIMEText(self.auto_reply_message, 'plain'))
            
            # Encode message
            raw_message = base64.urlsafe_b64encode(reply.as_bytes()).decode('utf-8')
            
            # Send reply
            send_result = self.service.users().messages().send(
                userId='me',
                body={
                    'raw': raw_message,
                    'threadId': original_email['thread_id']
                }
            ).execute()
            
            logger.info(f"Auto-reply sent for email: {original_email['subject']}")
            return True
            
        except Exception as error:
            logger.error(f"Error sending reply: {error}")
            return False
    
    def mark_as_read(self, message_id):
        """Mark an email as read."""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            logger.info(f"Marked email {message_id} as read")
        except Exception as error:
            logger.error(f"Error marking email as read: {error}")
    
    def process_emails(self):
        """Main function to process unread emails and send auto-replies."""
        logger.info("Starting email processing...")
        
        unread_emails = self.get_unread_emails()
        
        for message in unread_emails:
            message_id = message['id']
            
            # Get email details
            email_details = self.get_email_details(message_id)
            if not email_details:
                continue
            
            logger.info(f"Processing email from: {email_details['sender']}, Subject: {email_details['subject']}")
            
            # Check if auto-reply is needed
            if self.should_auto_reply(email_details):
                # Send auto-reply
                if self.send_reply(email_details):
                    logger.info(f"Auto-reply sent to: {email_details['sender']}")
                    
                    # Mark as processed for current session
                    self.processed_emails.add(message_id)
                    
                    # Mark as read - this prevents future processing
                    self.mark_as_read(message_id)
            else:
                logger.info(f"No auto-reply needed for email from: {email_details['sender']}")
                # Still mark as processed for current session to avoid checking again
                self.processed_emails.add(message_id)
    
    def run_continuously(self, check_interval=300):
        """Run the bot continuously, checking for new emails every few minutes."""
        logger.info(f"Starting Gmail bot - checking every {check_interval} seconds")
        
        while True:
            try:
                self.process_emails()
                logger.info(f"Sleeping for {check_interval} seconds...")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as error:
                logger.error(f"Unexpected error: {error}")
                logger.info("Retrying in 60 seconds...")
                time.sleep(60)
    
    def run_once(self):
        """Run the bot once (useful for testing)."""
        logger.info("Running Gmail bot once...")
        self.process_emails()
        logger.info("Gmail bot run completed")

def main():
    """Main function to run the Gmail bot automatically."""
    bot = GmailBot()
    
    # Authenticate with Gmail
    bot.authenticate()
    
    # Configuration is already set in __init__, but you can override here if needed:
    # bot.trigger_keywords = ['BOLETA - MEDILESER S.A.C', 'CONTRATO DE TRABAJO - MEDILESER S.A.C']
    # bot.trigger_senders = ['documentoslaborales@medileser.com.pe']
    # bot.auto_reply_message = """Recibi conforme"""
    
    logger.info("Gmail Bot Configuration:")
    logger.info(f"Trigger keywords: {bot.trigger_keywords}")
    logger.info(f"Monitoring specific senders: {bot.trigger_senders if bot.trigger_senders else 'All senders'}")
    logger.info("Starting automated monitoring...")
    
    # Run continuously with 5-minute intervals (300 seconds)
    bot.run_continuously(check_interval=300)

if __name__ == "__main__":
    main()