#!/bin/bash

id=`ps aux|grep helmet_serv|grep -v grep|awk '{print $2}'`
echo $id
if [ -n "$id" ]
then
kill -9 $id
sleep 1
fi

nohup python helmet_serv.py &
