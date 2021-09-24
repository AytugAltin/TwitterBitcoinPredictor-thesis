import subprocess
import time, datetime
import os
import csv
from multiprocessing import Pool

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]



def run_queries(Dates):
	attemps = 0
	for day in Dates:
		attemps += 1
		if len(str(day)) == 1:
			call = 'python Exporter.py --lang en --querysearch "bitcoin" --since '+ year + '-' + month + '-0' + str(day) + ' --until '+ year + '-' + month + '-0' +str(day+1)
			output_file = "{}-{}-0{}.csv".format(year, month, str(day))
		if len(str(day)) == 2:
			call = 'python Exporter.py --lang en --querysearch "bitcoin" --since '+ year +'-' + month + '-' + str(day) + ' --until '+ year + '-' + month + '-' +str(day+1)
			output_file = "{}-{}-{}.csv".format(year, month, str(day))

		subprocess.call(call, shell=True)

		tweets = []
		
		with open(output_file, encoding="utf8") as csvfile: 
			file = csv.reader(csvfile, delimiter=';') 
		
			for row in file:
				tweets.append(row[1])

		csvfile.close()

		output_file_incomplete = "To process/{}-{}-{}.csv".format(year, month, str(day))
		
		if tweets[-1].split(" ")[1] == "00:00:00":
			os.rename(output_file, folder_to_save + output_file)
		else:
			#os.remove(output_file)
			os.rename(output_file, output_file_incomplete)		
			#Dates.append(day)
		time.sleep(30)

#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
#python Exporter.py --lang en --querysearch "bitcoin" --since 2018-11-30 --until 2018-12-01

#DATES_WITH_ERRORS_AUG = [2,9,12,13,15,26,28] 
#DATES_WITH_ERRORS_SEP = [7,14,16,23,24,30]
#DATES_WITH_ERRORS_OCT = [3,6,7,22]
#DATES_WITH_ERRORS_NOV = [9,26,27]
#DATES_WITH_ERRORS_DEC = [2,5]
#DATES_WITH_ERRORS_JAN = [3,7,26]
#DATES_WITH_ERRORS_FEB = [18,19,24,25,26]
#DATES_WITH_ERRORS_MARS = [2,3,4,6,7]
#DATES_WITH_ERRORS_APRIL = [3]
#DATES_WITH_ERRORS_MAY = [22]
#DATES_WITH_ERRORS_JUN = [16,25]


#This is the information that you can change to run diferent queries!

year = "2019"
month = "01"
folder_to_save = 'Raw/Enero 2019/'
Dates = [2,3,5,6,7,8,10,12,14,15,17,18,19,21,22,23]
processes = 8

query = list(chunks(Dates, int((len(Dates)/processes))))

if __name__ == '__main__':
	with Pool(processes=processes) as pool:
		if processes == 1:
			pool.map(run_queries, [query[0]])
		if processes == 2:
			pool.map(run_queries, [query[0], query[1]])
		if processes == 3:
			pool.map(run_queries, [query[0], query[1], query[2]])
		if processes == 4:
			pool.map(run_queries, [query[0], query[1], query[2], query[3]])
		if processes == 5:
			pool.map(run_queries, [query[0], query[1], query[2], query[3], query[4]])
		if processes == 6:
			pool.map(run_queries, [query[0], query[1], query[2], query[3], query[4], query[5]])
		if processes == 7:
			pool.map(run_queries, [query[0], query[1], query[2], query[3], query[4], query[5], query[6]])
		if processes == 8:
			pool.map(run_queries, [query[0], query[1], query[2], query[3], query[4], query[5], query[6], query[7]])
		if processes == 9:
			pool.map(run_queries, [query[0], query[1], query[2], query[3], query[4], query[5], query[6], query[7], query[8]])
		if processes == 10:
			pool.map(run_queries, [query[0], query[1], query[2], query[3], query[4], query[5], query[6], query[7], query[8], query[9]])
		if processes > 10:
			print('Number of processes too big!')