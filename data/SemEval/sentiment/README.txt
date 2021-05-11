SemEval-2016 Task 6 Stance and Sentiment Dataset
Raw Sentiment Annotations
June 2019
Copyright (C) 2019 National Research Council Canada (NRC)
----------------------------------------------------------------


Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)


Terms of Use: 
-------------------------------------------------

1. If you use this dataset, cite the paper below:

Mohammad, S., Sobhani, P., Kiritchenko, S. (2017). Stance and Sentiment in Tweets. ACM Transactions on Internet Technology, 17(3), 2017.

2. Do not redistribute the data. Direct interested parties to this page: http://www.saifmohammad.com/WebPages/StanceDataset.htm

3. National Research Council Canada (NRC) disclaims any responsibility for the use of the dataset and does not provide technical support. However, the contact listed above will be happy to respond to queries and clarifications.


The file contains raw sentiment annotations for tweets used in SemEval-2016 Task 6 'Detecting Stance in Tweets'. The sentiment annotations  were not part of the SemEval-2016 shared task, but are made available for future research. Details about this dataset are available in the following paper:

Mohammad, S., Sobhani, P., Kiritchenko, S. (2017). Stance and Sentiment in Tweets. ACM Transactions on Internet Technology, 17(3), 2017.


************************************************
File Format
************************************************

Tha annotation file has the following format:
<Worker ID>,<Instance ID>,<Tweet>,<Sentiment>

where
<Worker ID> is an ID for a crowd worker; to preserve privacy, we replaced the CrowdFlower worker IDs with sequential IDs, but kept the same ID for all annotations performed by a particular worker;
<Instance ID> is an ID for an annotated tweet; the IDs are the same that were used in the SemEval-2016 Task 6 training and test datasets;
<Tweet> is a tweet text;
<Sentiment> is a worker's answer to the following question: 'What kind of language is the speaker using?' 

The possible answers are:
1. the speaker is using positive language, for example, expressions of support, admiration, positive attitude,
forgiveness, fostering, success, positive emotional state (happiness, optimism, pride, etc.)
2. the speaker is using negative language, for example, expressions of criticism, judgment, negative attitude,
questioning validity/competence, failure, negative emotional state (anger, frustration, sadness,
anxiety, etc.)
3. the speaker is using expressions of sarcasm, ridicule, or mockery
4. the speaker is using positive language in part and negative language in part
5. the speaker is neither using positive language nor using negative language.



************************************************
More Information
************************************************


Mohammad, S., Sobhani, P., Kiritchenko, S. (2017). Stance and Sentiment in Tweets. ACM Transactions on Internet Technology, 17(3), 2017.



************************************************
CONTACT INFORMATION
************************************************
Saif M. Mohammad
Senior Research Officer, National Research Council Canada
email: saif.mohammad@nrc-cnrc.gc.ca
phone: +1-613-993-0620
