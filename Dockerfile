FROM tensorflow/tensorflow 

WORKDIR /sudoku-ml
ADD . /sudoku-ml

RUN mkdir -p /log_dir/
RUN mkdir -p /models/

RUN pip install https://github.com/cloudmercato/sudoku-game/archive/refs/heads/master.zip
RUN python setup.py develop

VOLUME /log_dir/
VOLUME /models/
VOLUME /datasets/

CMD ["sudoku-ml", "--log-dir=/log_dir/", "--model-save-file=/models/current.h5", "--tf-profiler-port=6007"]

EXPOSE 6006/TCP
EXPOSE 6007/TCP
