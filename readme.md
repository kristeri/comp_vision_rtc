

## Installation

Install dependencies

```
pip install -r requirements.txt
```

Install PyTorch libraries based on the testing environment. Notice the correct CUDA version.
https://pytorch.org/get-started/locally/

For example if the testing hardware is compatible with CUDA 11.7:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```
Testing the application outside localhost requires an SSL certificate for the WebRTC functionality. This applies to e.g. if the web server is accessed with another device in a local network. This can be done by generating cert.pem and key.pem with [OpenSSL](https://www.openssl.org/):

```
openssl req -newkey rsa:2048 -x509 -nodes -keyout key.pem -new -out cert.pem -config req.cnf -sha256 -days 3650
```

## Running the application

Start the server with:

```
python server.py --cert_file="cert.pem" --key_file="key.pem"
```

Access the application with browser:

https://localhost:8080/

If accessing the web app with another device in local network:

https://LOCAL_IP:8080/

where LOCAL_IP is the IP address of the server running the app.

