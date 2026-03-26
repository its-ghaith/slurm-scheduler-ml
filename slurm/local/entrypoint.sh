#!/bin/bash
set -euo pipefail

ROLE="${1:-controller}"

mkdir -p /run/munge /var/log/munge /var/log/slurm /var/spool/slurmctld /var/spool/slurmd
touch /var/log/munge/munged.log
chown -R munge:munge /run/munge /etc/munge
chown -R root:root /var/log/munge
chmod 0700 /run/munge
chmod 0755 /var/log/munge
chmod 0600 /var/log/munge/munged.log
chmod 0400 /etc/munge/munge.key

/usr/sbin/munged --force --log-file=/var/log/munge/munged.log

if [ "${ROLE}" = "controller" ]; then
  echo "Starting slurmctld..."
  exec /usr/sbin/slurmctld -D -vvv
fi

echo "Starting slurmd..."
exec /usr/sbin/slurmd -D -vvv
