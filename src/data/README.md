# Datasets

## File Structure

This directory should contain the datasets. Due to their large size, the full datasets are .gitignored, but sample datasets have been included. You should download and extract the full datasets such that the folder structure looks like the folowing:


```bash
data/
	README.md  # this file
	dmoz/
		URL Classification_sample.csv
		URL Classification.csv
	phishing/
		phishing_dataset_sample.csv
		phishing_dataset.csv
	webkb/
		course/
			cornell/
				http/^^cs.cornell.edu^Info^Courses^Current^CS415^CS414
				# more files with the url as filename
			# misc, texas, washington, wisconsin
		# department, faculty, other, project, staff, student
	webkb_sample/	
		# similar structure as above
```


## Obtaining the Datasets

The datasets can be obtained from:

* ILP 98 WebKB: <http://www.cs.cmu.edu/~webkb/>
* Phising etc.: <https://www.unb.ca/cic/datasets/url-2016.html>
* DMOZ - Open Directory Project Web (Kaggle URL Classification): <https://www.kaggle.com/shawon10/url-classification-dataset-dmoz>


## Additional Datasets

* Website Phising Data Set (Does not contain actual URL): <http://archive.ics.uci.edu/ml/datasets/Website+Phishing>
* ANT: <https://ant.isi.edu/datasets/>
