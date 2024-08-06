FROM postgres:latest
RUN apt-get update && \
    apt-get install -y build-essential postgresql-server-dev-all git
RUN git clone https://github.com/pgvector/pgvector.git /usr/src/pgvector && \
    cd /usr/src/pgvector && \
    make && \
    make install
RUN apt-get remove -y build-essential postgresql-server-dev-all git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /usr/src/pgvector 
