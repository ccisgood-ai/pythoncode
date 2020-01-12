def saver(userid_list, names_list,
          comments_list, created_time_list, course_times_list, voteup_list,
          rating_list, conn, cursor):
    '''
    @description: Save the comments info to mysql
    @param All field in mysql; {"conn,cursor": the code to use mysql}
    @return: None
    '''
    # saving to database
    for i in range(len(names_list)):
        userid = userid_list[i]
        author_name = names_list[i]
        comments = comments_list[i]
        created_time = created_time_list[i]
        course_times = course_times_list[i]
        voteup = voteup_list[i]
        rating = rating_list[i]
        line = [
            userid, author_name, comments,
            created_time, course_times, voteup, rating
        ]
        # print(line)
        insert_sql = """
                        insert into test1(userid,author_name, comments, created_time, course_times, voteup, rating)
                        values(%s,%s,%s,%s,%s,%s,%s)
                     """

        cursor.execute(
            insert_sql,
            (userid, author_name,comments, created_time, course_times, voteup, rating))

        conn.commit()
