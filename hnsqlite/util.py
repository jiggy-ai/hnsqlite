import hashlib
import re

def md5_file(filename):
    with open(filename, 'rb') as f:
        md5 = hashlib.md5()
        while True:
            data = f.read(1024*1024)
            if not data:
                break
            md5.update(data)
        md5sum = md5.hexdigest()            
    return md5sum


# Use DNS naming rules for convenience as they prohibit special characters, underscores, spaces, etc.
ALLOWED_NAME = re.compile("(?!-)[A-Z\d-]{1,63}(?<!-)$", re.IGNORECASE)   # REGEX for valid Name, based on DNS name component requirements

def _is_valid_namestr(checkstr, display):
    """
    validate the checkstring is a valid name per DNS name rules.
    raises ValueError if the checkstring is invalid.
    Will refer to the name as type "display" in the error messages to accomodate different field names using this validator
    """
    if len(checkstr) > 64 or not checkstr:
        raise ValueError('"%s" is an invalid %s. It must not be empty and is limited to 64 characters.' % (checkstr, display))
    elif not bool(ALLOWED_NAME.match(checkstr)):
        raise ValueError('"%s" is an invalid %s. It can only contain letters, numbers, and hyphens, and must not start or end with a hyphen.' % (checkstr, display))

