# Datasets

## File Structure

This directory should contain the datasets. Due to their large size, the full datasets are .gitignored, but sample datasets have been included. You should download and extract the full datasets and rename them if necessary such that the folder structure looks like the folowing:


```bash
data/
	README.md  # this file
	dmoz/
		URL Classification_sample.csv
		URL Classification.csv
	phishing/
		benign_dataset_sample.csv
		benign_dataset.csv
		phishing_dataset_sample.csv
		phishing_dataset.csv
		phishing_extra_sample.csv
		phishing_extra.csv
	webkb/
		course/
			cornell/
				http/^^cs.cornell.edu^Info^Courses^Current^CS415^CS414
				# more files with the url as filename
			# misc, texas, washington, wisconsin
		# department, faculty, other, project, staff, student
```


## Obtaining the Datasets

The datasets can be obtained from:

* ILP 98 WebKB: <http://www.cs.cmu.edu/~webkb/>
* Phishing etc.: <https://www.unb.ca/cic/datasets/url-2016.html>
* Phishing extra: <https://www.phishtank.com/developer_info.php>
* DMOZ - Open Directory Project Web (Kaggle URL Classification): <https://www.kaggle.com/shawon10/url-classification-dataset-dmoz>
