FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    slurm-wlm \
    munge \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY slurm/local/etc/slurm.conf /etc/slurm/slurm.conf
COPY slurm/local/etc/gres.conf /etc/slurm/gres.conf
COPY slurm/local/munge/munge.key /etc/munge/munge.key
COPY slurm/local/entrypoint.sh /entrypoint.sh

RUN chmod 400 /etc/munge/munge.key \
    && chown -R munge:munge /etc/munge \
    && mkdir -p /var/log/slurm /var/spool/slurmctld /var/spool/slurmd \
    && chown -R root:root /var/log/slurm /var/spool/slurmctld /var/spool/slurmd \
    && chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
