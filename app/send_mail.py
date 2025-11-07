import msal
import requests



class SPCMailer:
    @staticmethod
    def send_notification_email_sync(to: str, subject: str, body: str):
        app = msal.ConfidentialClientApplication(
            CLIENT_ID,
            authority=AUTHORITY_URL,
            client_credential=CLIENT_SECRET_VALUE
        )
        scopes = ["https://graph.microsoft.com/.default"]
        result = app.acquire_token_for_client(scopes=scopes)
        if "access_token" not in result:
            raise Exception(f"Could not acquire access token: {result.get('error_description', result)}")
        access_token = result["access_token"]

        recip = [{'EmailAddress': {'Address': email.strip()}} for email in to.split(",") if email.strip()]

        email_msg = {
            'Message': {
                'Subject': subject,
                'Body': {'ContentType': 'Html', 'Content': body},
                'ToRecipients': recip,
                'From': {
                    'EmailAddress': {
                        'Address': EMAIL_SENDER,
                        'Name': EMAIL_SENDER_NAME
                    }
                }
            },
            'SaveToSentItems': 'true'
        }

        userId = EMAIL_SENDER
        endpoint = f'https://graph.microsoft.com/v1.0/users/{userId}/sendMail'
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.post(endpoint, headers=headers, json=email_msg)
        if not response.ok:
            raise Exception(response.text)
        return True

"""
if __name__ == "__main__":
    # Example usage
    print(
        send_notification_email_sync(
            to="divesha@spc.int",
            subject="Test Subject",
            body="This is the email body."
        )
    )
"""