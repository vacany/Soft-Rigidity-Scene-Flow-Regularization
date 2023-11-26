#!/bin/sh


#sshpass -p "Coldplay7" scp vacekpa2@login3.rci.cvut.cz:~/4D-RNSFP/scripts/log/jupyter-notebook-current.log jupyter-notebook-current.log
sshpass -p "Coldplay7" scp vacekpa2@login3.rci.cvut.cz:~/jupyter-notebook-current.log jupyter-notebook-current.log

server=`cat ~/jupyter-notebook-current.log | grep "Remote server"`
server=${server##* }
port=`cat ~/jupyter-notebook-current.log | grep "Remote port"`
port=${port##* }

link=`cat ~/jupyter-notebook-current.log | grep "or http://"`
#token=`echo $line | sed 's/.*or //'`
echo " "
echo $link
echo " "

sshpass -p "Coldplay7" ssh -X -N -L -o ServerAliveInterval=60 $port:$server:$port vacekpa2@login3.rci.cvut.cz

