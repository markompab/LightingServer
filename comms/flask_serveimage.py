import os

from flask import send_file, app, request, Flask

app = Flask(__name__)

@app.route('/lightserver/<im_dttm>')
def get_image(im_dttm):
    impath = "../images/{}.jpg".format(im_dttm)

    if(not os.path.exists(impath)):
        impath = "../images/na.jpg"

    '''
    if request.args.get('type') == '1':
       filename = 'ok.gif'
    else:
       filename = 'error.gif'
        '''
    #return send_file(filename, mimetype='image/gif')

    return send_file(impath, mimetype='image/jpg')


app.run()