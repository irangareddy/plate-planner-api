#!/bin/bash

echo "=== Poetry & Python Environment Diagnostics ==="

echo "Current date: $(date)"
echo

echo "1. Poetry version:"
poetry --version || { echo "Poetry not found!"; exit 1; }
echo

echo "2. Poetry virtualenv info:"
poetry env info
echo

echo "3. Python version and path:"
poetry run python --version
poetry run which python
echo

echo "4. Checking if you're on macOS and if Install Certificates.command exists:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    CERT_CMD="/Applications/Python 3.12/Install Certificates.command"
    if [ -f "$CERT_CMD" ]; then
        echo "Found: $CERT_CMD"
        echo "If you haven't run this, do so now to fix SSL issues."
    else
        echo "Install Certificates.command not found for Python 3.12."
        echo "If you installed Python via Homebrew, certs are handled differently."
    fi
else
    echo "Not on macOS, skipping this step."
fi
echo

echo "5. Checking certifi CA bundle location (used by requests/ssl):"
poetry run python -c "import certifi; print(certifi.where())"
echo

echo "6. Checking Python SSL default verify paths:"
poetry run python -c "import ssl; print(ssl.get_default_verify_paths())"
echo

echo "7. Testing HTTPS connectivity from Python:"
poetry run python -c "import urllib.request; print('OK' if urllib.request.urlopen('https://pypi.org').status==200 else 'FAIL')"
echo

echo "8. Attempting to download NLTK 'verbnet' corpus:"
poetry run python -c "import nltk; nltk.download('verbnet')"
echo

echo "9. Checking if 'verbnet' is now available:"
poetry run python -c "import nltk; print('Found:', nltk.data.find('corpora/verbnet'))"
echo

echo "=== Diagnostics Complete ==="
echo
echo "If you still see SSL or certificate errors above:"
echo "- On macOS: Run the Install Certificates.command for your Python version."
echo "- If using Homebrew Python, try: brew install certifi"
echo "- Ensure your NLTK_DATA environment variable is set if you use a custom location."
echo "- If behind a proxy/firewall, configure your proxy settings for Python and NLTK."
echo
echo "If 'verbnet' is found, you are ready to rerun your script!"
