#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""To use this script you can pass the following attributes:
       querysearch: a query text to be matched
          username: a username or a list of usernames (comma or space separated)
                    of a specific twitter account(s) (with or without @)
username-from-file: a file with a list of usernames,
             since: a lower bound date in UTC (yyyy-mm-dd)
             until: an upper bound date in UTC (yyyy-mm-dd) (not included)
              near: a reference location area from where tweets were generated
            within: a distance radius from "near" location (e.g. 15mi)
         toptweets: only the tweets provided as top tweets by Twitter (no parameters required)
         maxtweets: the maximum number of tweets to retrieve
              lang: the language of tweets
            output: a filename to export the results (default is "output_got.csv")

Examples:
# Example 1 - Get tweets by username
GetOldTweets3 --username "barackobama" --maxtweets 1

# Example 2 - Get tweets by several usernames (use multiple --username options
# or a comma/space separated list):
GetOldTweets3 --username "BarackObama,AngelaMerkeICDU" --username "WhiteHouse" --maxtweets 10

# Example 3 - Get top tweets from users specified in files and also specific users:
GetOldTweets3 --usernames-from-file userlist.txt --usernames-from-file additinal_list.txt --username "barackobama whitehouse" --toptweets

# Example 4 - Get tweets by query search
GetOldTweets3 --querysearch "europe refugees" --maxtweets 10

# Example 5 - Get tweets by username and bound dates (until date is not included)
GetOldTweets3 --username "barackobama" --since 2015-09-10 --until 2015-09-12 --maxtweets 10

# Example 6 - Get the last 10 top tweets by username
GetOldTweets3 --username "barackobama" --maxtweets 10 --toptweets
"""

import os, sys, re, getopt
if sys.version_info[0] < 3:
    raise Exception("Python 2.x is not supported. Please upgrade to 3.x")

import GetOldTweets3 as got
import time
import winsound

duration = 1000  # millisecond
freq = 440  # Hz

then = time.time() #Time before the operations start

def main(argv):
    if len(argv) == 0:
        print('You must pass some parameters. Use \"-h\" to help.')
        return

    if len(argv) == 1 and argv[0] == '-h':
        print(__doc__)
        return

    try:
        opts, args = getopt.getopt(argv, "", ("querysearch=",
                                              "username=",
                                              "usernames-from-file=",
                                              "since=",
                                              "until=",
                                              "near=",
                                              "within=",
                                              "toptweets",
                                              "maxtweets=",
                                              "lang=",
                                              "output=",
                                              "debug"))

        tweetCriteria = got.manager.TweetCriteria()
        outputFileName = "output_got.csv"

        debug = False
        usernames = set()
        username_files = set()
        for opt, arg in opts:
            if opt == '--querysearch':
                tweetCriteria.querySearch = arg

            elif opt == '--username':
                usernames_ = [u.lstrip('@') for u in re.split(r'[\s,]+', arg) if u]
                usernames_ = [u.lower() for u in usernames_ if u]
                usernames |= set(usernames_)

            elif opt == '--usernames-from-file':
                username_files.add(arg)

            elif opt == '--since':
                tweetCriteria.since = arg
                outputFileName = arg + '.csv'
                
            elif opt == '--until':
                tweetCriteria.until = arg

            elif opt == '--near':
                tweetCriteria.near = '"' + arg + '"'

            elif opt == '--within':
                tweetCriteria.within = arg

            elif opt == '--toptweets':
                tweetCriteria.topTweets = True

            elif opt == '--maxtweets':
                tweetCriteria.maxTweets = int(arg)

            elif opt == '--lang':
                tweetCriteria.lang = arg

            elif opt == '--output':
                outputFileName = arg

            elif opt == '--debug':
                debug = True

        if debug:
            print(' '.join(sys.argv))
            print("GetOldTweets3", got.__version__)

        if username_files:
            for uf in username_files:
                if not os.path.isfile(uf):
                    raise Exception("File not found: %s"%uf)
                with open(uf) as f:
                    data = f.read()
                    data = re.sub('(?m)#.*?$', '', data)  # remove comments
                    usernames_ = [u.lstrip('@') for u in re.split(r'[\s,]+', data) if u]
                    usernames_ = [u.lower() for u in usernames_ if u]
                    usernames |= set(usernames_)
                    print("Found %i usernames in %s" % (len(usernames_), uf))

        if usernames:
            if len(usernames) > 1:
                tweetCriteria.username = usernames
                if len(usernames)>20 and tweetCriteria.maxTweets > 0:
                    maxtweets_ = (len(usernames) // 20 + (len(usernames)%20>0)) * tweetCriteria.maxTweets
                    print("Warning: due to multiple username batches `maxtweets' set to %i" % maxtweets_)
            else:
                tweetCriteria.username = usernames.pop()

        outputFile = open(outputFileName, "w+", encoding="utf8")
        outputFile.write('username;date;retweets;favorites;text;geo;mentions;hashtags;id;permalink;to\n')

        cnt = 0
        def receiveBuffer(tweets):
            nonlocal cnt

            for t in tweets:
                data = [t.username,
                    t.date.strftime("%Y-%m-%d %H:%M:%S"),
                    t.retweets,
                    t.favorites,
                    '"'+t.text.replace('"','""')+'"',
                    t.geo,
                    t.mentions,
                    t.hashtags,
                    t.id,
                    t.permalink,
                    t.to or '']
                data[:] = [i if isinstance(i, str) else str(i) for i in data]
                outputFile.write(';'.join(data) + '\n')

            outputFile.flush()
            cnt += len(tweets)

            if sys.stdout.isatty():
                print("\rSaved %i"%cnt, end='', flush=True)
            else:
                print(cnt, end=' ', flush=True)

        print("Downloading tweets...")
        got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer, debug=debug)
        
    except getopt.GetoptError as err:
        print('Arguments parser error, try -h')
        print('\t' + str(err))

    except Exception as err:
        print(str(err))

    finally:
        if "outputFile" in locals():
            outputFile.close()
            print()
            print('Done. Output file generated "%s".' % outputFileName)

if __name__ == '__main__':
    main(sys.argv[1:])


now = time.time() #Time after it finished
print("It took: ", now-then, " seconds")
winsound.Beep(freq, duration)
