FROM ricardojmmaia/mpspdz

USER root

RUN apt-get install -y python3-pandas
RUN apt-get install -y  python3-numpy
RUN apt-get install -y python3-sklearn

RUN  chmod 777 -R /opt/app/MP-SPDZ/
ENV TZ=America/Indiana/Indianapolis
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /opt/app/MP-SPDZ/

RUN Scripts/setup-ssl.sh 3
RUN c_rehash Player-Data

ADD ml.py /opt/app/MP-SPDZ/Compiler/
ADD mpc_noise.py /opt/app/MP-SPDZ/Compiler/
ADD Step1_Preprocess.py /opt/app/MP-SPDZ/
ADD Step5_classification.py /opt/app/MP-SPDZ/
ADD Step3_LR_training.mpc /opt/app/MP-SPDZ/Programs/Source/lr_training.mpc
ADD run_all_steps.sh /opt/app/MP-SPDZ/

RUN /opt/app/MP-SPDZ/compile.py -R 64 -Y lr_training 832 831 1784 300 128 1 1
