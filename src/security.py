import ipaddress
from urllib.parse import urlparse
import socket
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_url(url):
    """
    Validates a URL to prevent SSRF and other attacks.
    Args:
        url (str): The URL to validate.
    Returns:
        bool: True if the URL is valid and safe, False otherwise.
    """
    try:
        parsed_url = urlparse(url)
        
        # Check scheme
        if parsed_url.scheme not in ['http', 'https']:
            logger.warning(f"Invalid scheme for URL: {url}")
            return False
            
        # Check hostname
        hostname = parsed_url.hostname
        if not hostname:
            logger.warning(f"No hostname in URL: {url}")
            return False
            
        # Resolve IP address to check if it's local/private
        try:
            ip_address = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_address)
            
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                logger.warning(f"Blocked private/local IP access: {url} ({ip_address})")
                return False
                
        except socket.gaierror:
            logger.warning(f"Could not resolve hostname: {hostname}")
            return False
            
        return True

    except Exception as e:
        logger.error(f"Url validation error: {e}")
        return False
