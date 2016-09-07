# Author: Julien Hoachuck
# Copyright 2015, Julien Hoachuck, All rights reserved.
#!/bin/bash
file1="0"
file2="0"

mongo spearmint --eval "db.dropDatabase()"

#find //var/lib/mongodb/ -type f -name "spearmint.*" 2>/dev/null | grep -q . && file1=1
#if [ $file1 -eq 1 ]; then
#	rm //var/lib/mongodb/spearmint.*
#	echo "Removed spearmint.*"
#fi

#find //var/lib/mongodb/ -maxdepth 1 -type f -name "mongod.lock" 2>/dev/null | grep -q . && file2=1
#if [ $file2 -eq 1 ]; then
#	rm //var/lib/mongodb//mongod.lock
#	echo "Removed mongod.lock"
#fi

#killall mongod

