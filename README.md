# hummingRecognition
Recognize song by humming into application

Setup:
1. Install the Python development environment on your system(pip is included in this): https://www.python.org/ftp/python/3.8.5/python-3.8.5-amd64.exe
2. Create a virtual environment
***Python virtual environments are used to isolate package installation from the system.
	python -m venv --system-site-packages .\venv
***Activate the virtual environment: .\venv\Scripts\activate

***Install packages within a virtual environment without affecting the host system setup. Start by upgrading pip: pip install --upgrade pip

3. Install the TensorFlow pip package
   For GPU and CPU support:pip install --upgrade tensorflow
   For CPU only (for office laptop):pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow_cpu-2.3.0-cp38-cp38-win_amd64.whl

   Verify the install: python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

