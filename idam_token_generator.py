"""
Utility to generate the IDAM access token for LLM service app
"""
import os
from datetime import datetime, timedelta, timezone
import base64
import uuid
import json
import jwt
import requests


CACHE_FILE_PATH = os.path.join(os.path.dirname(__file__), "__idamcache__")


class IDAMTokenGenerator:
    """
    This class generates IDAM tokens.
    """
    

    def __init__(self, idam_token_endpoint, idam_app_client_id, idam_app_client_secret, lmaas_idam_app_audience):
        self.idam_token_endpoint = idam_token_endpoint
        self.idam_app_client_id = idam_app_client_id
        self.idam_app_client_secret = idam_app_client_secret
        self.lmaas_idam_app_audience = lmaas_idam_app_audience


    def get_unique_number(self):
        """
        This function get unique number.

        Returns:
            UUID : Unique number.
        """
        return uuid.uuid4()


    def get_access_token(self):
        """
        Input : IDAM Application Client ID and Client secret

        Returns: Access token
        """
        payload = f'grant_type=client_credentials&client_id={self.idam_app_client_id}&client_secret={self.idam_app_client_secret}&scope={self.get_unique_number()}'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        try:
            response_data = requests.request("POST", self.idam_token_endpoint, \
                headers=headers, data=payload, verify=False, timeout=60)
            response_data.raise_for_status()
            if response_data is not None and response_data.ok:
                access_token = response_data.json()['access_token']
                print("IDAM Access Token is generated")
                return access_token
        except requests.RequestException as e:
            print("Error sending POST request to %s: %s", self.idam_token_endpoint, e)
            raise


    def get_exchange_token(self, access_token):
        """
        Input : IDAM access token
        
        Returns: Exchange token
        """
        payload = f'grant_type=token_exchange&token={access_token}&audience={self.lmaas_idam_app_audience}&scope={self.get_unique_number()}'
        basic_auth_creds = f"{self.idam_app_client_id}:{self.idam_app_client_secret}"
        basic_auth_creds = base64.b64encode(basic_auth_creds.encode()).decode("ascii")
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': f'Basic {basic_auth_creds}'
        }
        try:
            response_data = requests.request("POST", self.idam_token_endpoint, \
                headers=headers, data=payload, verify=False, timeout=60)
            response_data.raise_for_status()
            if response_data is not None:
                exchange_token = response_data.json()['access_token']
                print("IDAM Exchange Access Token is generated")
                return exchange_token
        except requests.RequestException as e:
            print("Error sending POST request to %s for exchange token: %s", self.idam_token_endpoint, e)
            raise


    def is_valid_token(self, token):
        """
        This function checks if a JWT token is expired.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        try:
            if token is not None and token != "":
                # Use PyJWT to decode the token
                decoded_token = jwt.decode(token, options={"verify_signature": False})
                exp_time = decoded_token.get('exp')
                print("Expiry_time: %s", exp_time)
                if exp_time:
                    current_time = datetime.now(timezone.utc)
                    exp_time = datetime.fromtimestamp(exp_time, tz=timezone.utc)
                    # exp_time = datetime.datetime.fromtimestamp(exp_time, datetime.UTC)
                    print("exp_time : %s", exp_time)
                    # current_time = datetime.datetime.now(datetime.UTC)
                    print("current_time : %s", current_time)
                    if exp_time - current_time > timedelta(minutes=1):
                        print("JWT token is VALID")
                        return True
                    else:
                        print("JWT token is NOT VALID")
                        return False
                else:
                    print("JWT token is NOT VALID")
                    return False
            else:
                print("JWT token is NOT VALID")
                return False
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as ex:
            # Handle decode errors (e.g., expired or invalid token)
            print("JWT token expired. Please re-create. %s", ex)
            return False
        except (TypeError, ValueError, KeyError) as e:
            # Handle specific potential exceptions
            print("A specific error occurred: %s", e)
            return False


    def get_idam_token(self):
        """
        This function get the token. if a JWT token is expired.
        Create new access token.

        Returns:
            string : Acess token.
        """
        token = self.get_token_from_cache()
        if not self.is_valid_token(token):
            print("Generating new token.")
            try:
                token = self.get_access_token()
                if token is not None:
                    token = self.get_exchange_token(token)
                self.write_token_to_cache(token)
                return token
            except requests.exceptions.HTTPError as e:
                print("Failed to get the IDAM Access Token. Due to HTTP error occurred: %s", e)
            except requests.RequestException as e:
                print("Failed to get the IDAM Access Token. Due to Request error occurred: %s", e)
        else:
            print("Reusing existing valid token.")
            return token


    def write_token_to_cache(self, token):
        """
        Write to cache
        """
        key_str = self.idam_token_endpoint + self.idam_app_client_id
        key_val = base64.b64encode(key_str.encode()).decode("ascii")
        file_content = {}
        with open(CACHE_FILE_PATH, "r", encoding="utf-8") as file:
            file_content = json.load(file)
        file_content[key_val] = token
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as file:
            json.dump(file_content, file)
    
    
    def get_token_from_cache(self):
        """
        Read from cache
        """
        key_str = self.idam_token_endpoint + self.idam_app_client_id
        key_val = base64.b64encode(key_str.encode()).decode("ascii")
        file_content = {}
        file_content[key_val] = ""
        if not os.path.exists(CACHE_FILE_PATH):
            with open(CACHE_FILE_PATH, "w", encoding="utf-8") as file:
                json.dump(file_content, file)
        else:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as file:
                file_content = json.load(file)
        return file_content.get(key_val)

