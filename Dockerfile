FROM public.ecr.aws/sam/build-python3.8

COPY requirements.txt .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

#COPY detect.py ${LAMBDA_TASK_ROOT}
#COPY models/ ${LAMBDA_TASK_ROOT}/models
#COPY utils/ ${LAMBDA_TASK_ROOT}/utils
