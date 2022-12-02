import requests
import pandas as pd
import json
from tqdm import tqdm
import re

query_url = "https://leetcode.com/graphql/"
# get query from leetcode website for the question list
question_list_query = {"query":"\n    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {\n  problemsetQuestionList: questionList(\n    categorySlug: $categorySlug\n    limit: $limit\n    skip: $skip\n    filters: $filters\n  ) {\n    total: totalNum\n    questions: data {\n      acRate\n      difficulty\n      freqBar\n      frontendQuestionId: questionFrontendId\n      isFavor\n      paidOnly: isPaidOnly\n      status\n      title\n      titleSlug\n      topicTags {\n        name\n        id\n        slug\n      }\n      hasSolution\n      hasVideoSolution\n    }\n  }\n}\n    ",
         "variables":{"categorySlug":"","skip":0,"limit":2500,"filters":{}}}
# use query to request
problems_list = requests.post(query_url, json=question_list_query).json()

# get problem data list
problems_data_list = problems_list.get('data').get('problemsetQuestionList').get('questions')
print("Problems' Number:", len(problems_data_list))

# get topic for each question from the topics list
def get_formatted_data(topics_list):
    # create an array to store topics
    topics = []
    for topic in topics_list:
        topics.append(topic.get('name'))
    return topics

# get slug from data list
def getSlugs(data_list):
    # create an array to store slugs
    slugs = []
    # go through all the data in data_list
    for data in data_list:
        slugs.append(data.get('titleSlug'))
    return slugs

# get topics from data list
def getTopics(data_list):
    # create an array to store topics
    topics = []
    # go through all the data in data_list
    for data in data_list:
        topics.append(get_formatted_data(data.get('topicTags')))
    return topics


# get query from leetcode website for the question details
def get_question_details(title_slug):
    row = []
    # create question details query for each title_slug
    question_details_query = {"operationName": "questionData", "variables": {"titleSlug": title_slug},
                              "query": "query questionData($titleSlug: String!) {\n  question(titleSlug: $titleSlug) {\n    questionId\n    questionFrontendId\n    boundTopicId\n    title\n    titleSlug\n    content\n    translatedTitle\n    translatedContent\n    isPaidOnly\n    difficulty\n    likes\n    dislikes\n    isLiked\n    similarQuestions\n    exampleTestcases\n    categoryTitle\n    contributors {\n      username\n      profileUrl\n      avatarUrl\n      __typename\n    }\n    topicTags {\n      name\n      slug\n      translatedName\n      __typename\n    }\n    companyTagStats\n    codeSnippets {\n      lang\n      langSlug\n      code\n      __typename\n    }\n    stats\n    hints\n    solution {\n      id\n      canSeeDetail\n      paidOnly\n      hasVideoSolution\n      paidOnlyVideo\n      __typename\n    }\n    status\n    sampleTestCase\n    metaData\n    judgerAvailable\n    judgeType\n    mysqlSchemas\n    enableRunCode\n    enableTestMode\n    enableDebugger\n    envInfo\n    libraryUrl\n    adminUrl\n    challengeQuestion {\n      id\n      date\n      incompleteChallengeCount\n      streakCount\n      type\n      __typename\n    }\n    __typename\n  }\n}\n"}

    # request data from the leetcode website
    question_details = requests.post(query_url, json=question_details_query).json().get('data').get('question')

    # load data stats
    stats = json.loads(question_details.get('stats'))

    # append every row data into row[]
    row.append(question_details.get('questionId'))
    row.append(question_details.get('title'))
    row.append(question_details.get('titleSlug'))
    row.append(question_details.get('content'))
    row.append(question_details.get('difficulty'))
    row.append(stats.get('acRate'))
    row.append(question_details.get('companyTagStats'))
    row.append(question_details.get('likes'))
    row.append(question_details.get('dislikes'))
    row.append(question_details.get('hints'))
    row.append(json.loads(question_details.get('similarQuestions')))
    row.append(stats.get('totalAcceptedRaw'))
    row.append(stats.get('totalSubmissionRaw'))

    return row

def create_dataframe(slugs, topics):
    data_raw = []
    for i in tqdm(range(len(slugs))):
        temp_data = get_question_details(slugs[i])
        temp_data.append(topics[i])
        data_raw.append(temp_data)
    df = pd.DataFrame(data_raw)
    return df

slugs, topics = getSlugsandTopics(problems_data_list)
df = create_dataframe(getSlugs(problems_data_list), topics)

columns = [
    'question_id', 'question_title',
    'question_slug',  'question_text',
    'difficulty', 'success_rate',
    'company_tags', 'likes', 'dislikes', 'hints',
    'similar_questions', 'total_accepted',
    'total_submissions', 'topic_tagged_text'
    ]
df.columns = columns

df.to_csv('dataset/data_raw.csv', index = None)

