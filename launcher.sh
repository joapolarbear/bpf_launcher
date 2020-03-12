#!/usr/bin/expect 
set ip [lindex $argv 0]
set COMMAND [lindex $argv 1]
set PROMPT ":~*"
set timeout 180

spawn ssh root@${ip}
expect "${PROMPT}"
send "${COMMAND}\r"
expect "${PROMPT}"
send "exit\r"
expect eof 