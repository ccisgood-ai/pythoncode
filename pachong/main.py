import pymysql
from selenium import webdriver
from pachong.parser import get_courses_url, paser_comments
from pachong.saver import saver

def main():
    '''
    @description: This is the main function to set the database info and load the webdriver, then start the crawler
    '''
    conn = pymysql.connect(
        host='127.0.0.1', user='root', password='admin', database='scratch_test',charset='utf8', use_unicode=True,db='mysql')
    cursor = conn.cursor()
    driver = webdriver.Chrome(executable_path="F:\Python\chromedriver.exe")
    driver.maximize_window()
    # category list from mooc category
    # category_list = [
    #     'computer', 'foreign-language', 'psychology', 'ECO', 'management',
    #     'law', 'literature', 'historiography', 'philosophy', 'engineering',
    #     'science', 'biomedicine', 'agriculture', 'art-design',
    #     'teaching-method'
    # ]


    url = 'https://www.icourse163.org/course/BIT-47004'

    try:
        userid_list, names_list, comments_list, created_time_list, course_times_list, voteup_list, rating_list = paser_comments(
            url,driver)

        saver(userid_list,
              names_list, comments_list, created_time_list,
              course_times_list, voteup_list, rating_list, conn,
              cursor)

    except Exception:
        print("出现异常")


    driver.quit()
    conn.close()

    print("\nALL Done...")


if __name__ == "__main__":
    main()


