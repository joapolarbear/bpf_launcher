#!/usr/bin/expect 
set usrname [lindex $argv 0]
set PROMPT [lindex $argv 1]
set ip [lindex $argv 2]
set COMMAND [lindex $argv 3]
set timeout 180

spawn ssh ${usrname}@${ip}
expect "${PROMPT}"
send "${COMMAND}\r"
expect "${PROMPT}"
send "exit\r"
expect eof 