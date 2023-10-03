# Object detection webserver for the raspberry pi

The performance of the RPi4 is good enough to allow real time inference for object detection (See tutorial [here](https://pytorch.org/tutorials/intermediate/realtime_rpi.html)).
Based on this pytorch implementation, I build a simple django webserver, to visualize the captured images of the camera and the predicted object classes.

## Installation instructions

The instructions are provided for the Raspberry Pi OS.
After cloning the git repo, install the apache webserver (Full explanation [here](https://pimylifeup.com/raspberry-pi-apache/)):

`sudo apt install apache2 -y`

Next configure Apache for django as explained [here](https://pimylifeup.com/raspberry-pi-django/).
Especially important is the configuration for the `/etc/apache2/sites-enabled/000-default.conf` file. Make sure to replace the paths with the correct paths to the folder and pytheon virual environment. Here is an example config:

```
alias /static /home/pi/rpi-objectdetection-webserver/static
		<Directory /home/pi/rpi-objectdetection-webserver/static>
			LimitRequestBody 1024000 
			Require all granted
		</Directory>

		<Directory /home/pi/rpi-objectdetection-webserver/objectdetection>
			LimitRequestBody 1024000 
			<Files wsgi.py>
				Require all granted
			</Files>
		</Directory>

		WSGIDaemonProcess django python-path=/home/pi/rpi-objectdetection-webserver python-home=/home/pi/rpi-objectdetection-webserver/torch-django
		WSGIProcessGroup django
		WSGIScriptAlias / /home/pi/rpi-objectdetection-webserver/objectdetection/wsgi.py

```

After restarting apache (`sudo systemctl restart apache2`) install the python requirements in a new virtual environment:

```
python -m venv torch-django
source torch-django/bin/activate
pip install -r requirements.txt
```
Now everything is set up to start the server:

```
python manage.py migrate
python manage.py runserver
```

The webserver is now running at: [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
