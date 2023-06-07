import os
import io
from flask import Flask, jsonify, request, send_file, make_response
from flask_cors import CORS
import json
import time
import Diffuse
from PIL import Image
from gpt.Getprompt import TextProcessing_gpt, TextProcessing_T5
from img_to_3d import mesh_with_shap_e as shap
import pymysql
from rembg import remove
import time
import shutil
#from flask import abort
import jwt
import bcrypt
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity, get_jwt
from operator import itemgetter

app = Flask(__name__)
ID_seq = {}
# JWT 설정
app.config['JWT_SECRET_KEY'] = 'super-secret'
jwt = JWTManager(app)

# CORS 설정
CORS(app, supports_credentials=True)

@app.route('/call', methods= ['GET', 'POST'])
def call():
    con = pymysql.connect(host='127.0.0.1',port=3306, user='root',db='free3d', charset='utf8') 
    cur = con.cursor()
    item_list = ['item_id','ID','Image_path','_3d_path','model_name','category','price','description','title','is_sell','create_time','viewcount','downloadcount']
    if request.method == 'POST':
        get_type = request.args.get('type')
        print(get_type)
        
        if get_type == "Diffusion":
            prompt = request.form["text"]
            ID = request.form["ID"]
            model_name = request.form["model_name"]
            model = model_name.split('_')
            if model[0] == 'GPT':
                image = Diffuse.run(prompt)
                print("image_ready")
            else:
                image = Diffuse.run2(prompt)
                print("image_ready")
            if not os.path.exists('./static/'+ID):
                os.makedirs('./static/'+ID)
            rb_image = remove(image)
            if ID in ID_seq:
                seq = ID_seq.get(ID)
                seq += 1
                ID_seq[ID] = seq
            else:
                ID_seq[ID] = 0
            ret_name = ID+"/generated_img"+str(ID_seq[ID])+".png"
            print(ret_name)
            rb_image.save("./static/"+ret_name,'png')
            return ret_name
        
        elif get_type == "Modify":
            file = request.files['Image']
            ID = request.form["ID"]
            model_name = request.form["model_name"]
            file.save('./image.png') 
            prompt = request.form["text"]
            image = Diffuse.modify(prompt)
            print("image_ready")
            if not os.path.exists('./static/'+ID):
                os.makedirs('./static/'+ID)
            rb_image = remove(image)
            if ID in ID_seq :
                seq = ID_seq.get(ID)
                seq += 1
                ID_seq[ID] = seq
            else:
                ID_seq[ID] = 0
            ret_name = ID+"/generated_img"+str(ID_seq[ID])+".png"
            print(ret_name)
            rb_image.save("./static/"+ret_name,'png')
            return ret_name
        
        elif get_type == "3D":
            file = request.files['Image']
            ID = request.form["ID"]
            model_name = request.form["model_name"]
            file.save('./static/'+ID+'3d_image.jpg')
            shap.create_mesh('./static/'+ID+'3d_image.jpg',ID)
            return ID+"/generated_3d.obj"
        
        elif get_type == "Save":
            ID = request.form["ID"]
            model_name = request.form["model_name"]
            sql = "SELECT item_id FROM item WHERE ID=\'" +ID + "\'"
            cur.execute(sql)
            rows = cur.fetchall()
            print(rows)
            if len(rows)==0:
                num = 0
            else: 
                num = int(rows[-1][-1][-3:])
            num +=1
            item_id = str(num)
            if(len(item_id)==1):
                item_id = '00' + item_id
            elif(len(item_id)==2):
                item_id = '0' + item_id
            item_id = ID+item_id
            if not os.path.exists('./static/'+ID):
                os.makedirs('./static/'+ID)
            ret_name = "./static/"+ID+"/generated_img"+str(ID_seq[ID])+".png"
            Img = Image.open(ret_name)
            Img_path = ID+'/'+item_id+'image.png'
            _3d_path = ID+'/'+item_id+'3d.obj'
            Img.save('./static/'+Img_path)
            shutil.copy('./static/'+ID+'/generated_3d.obj','./static/'+_3d_path)
            sql = "INSERT INTO item(item_id, ID, model_name,Image_path,3d_path,is_sell) VALUES('" +item_id + "', '"+ ID + "', '"+ model_name + "', '" + Img_path +"', '" +_3d_path +"', 0)"
            cur.execute(sql)
            con.commit()
            return "OK"
        
        elif get_type == "Sell":
            item_id = request.form["item_id"]
            category = request.form["category"]
            price = request.form["price"]
            description = request.form["description"]
            title = request.form["title"]
            print([category,int(price),description,title,item_id])
            sql = "update item set category = %s, price= %s, description = %s, title = %s, is_sell = 1 where item_id = %s"
            cur.execute(sql,[category, int(price) , description, title, item_id])
            con.commit()
            return "OK"
        
        
        elif get_type == "Register":
            ID = request.form["ID"]
            password = request.form["password"]
            sql = "INSERT INTO user(ID, password) VALUES(\"" +ID + "\", \""+ password +"\");"
            print(sql)
            cur.execute(sql)
            con.commit()
            rows = cur.fetchall()
            print(rows)
            return "OK"
        
        
        elif get_type == "login":
            ID = request.form["ID"]
            password = request.form["password"]
            sql = "Select ID from user"
            cur.execute(sql)
            rows = cur.fetchall()
            rows = [list(rows[x]) for x in range(len(rows))]
            IDS = sum(rows, [])
            if ID in IDS:
                sql = "Select password from user where ID = \"" +ID+ "\";"
                cur.execute(sql)
                rows = cur.fetchall()
                rows = [list(rows[x]) for x in range(len(rows))]
                pass_db = rows[0][0]
                if pass_db == password:
                    print('jwt')
                    # JWT 생성
                    access_token = create_access_token(identity=ID)
                    response = make_response(jsonify({'access_token': access_token, 'email': ID}))
                    response.set_cookie('jwt_token', access_token, httponly=True)
                    return response
                else:
                    return jsonify({'msg': '아이디 또는 비밀번호가 일치하지 않습니다.'}), 401
            else:
                return jsonify({'msg': '아이디 또는 비밀번호를 확인하세요.'}), 401
            
        elif get_type == "discriminate" :
            text = request.form["text"]
            ID = request.form["ID"]
            model_name = request.form["model_name"]
            model = model_name.split('_')
            if model[0] == 'GPT':
                print("GPT")
                gp = TextProcessing_gpt(ner_model = 'DeveloperSejin/NER_for_furniture_3D_object_create',key='Your_key') #open_AI 키 입력
                answer = gp.getAnswer(prompt = text)
                del gp
            else:
                print("T5")
                t5 = TextProcessing_T5(ner_model = 'DeveloperSejin/NER_for_furniture_3D_object_create',model_name='DeveloperSejin/Fine_Tuned_Flan-T5-large_For_Describe_Furniture')
                answer = t5.getAnswer(prompt = text)
                del t5
            if(answer == -1):
                answer = {"recommend": "가구가 아닌 데이터는 금전적인 문제로 Fine tuning이 되지 않아 문장 생성을 도와드릴 수 없습니다. 기본 문장으로 생성을 진행해도 괜찮습니까?","detail": {"detail0": {"prompt": text, "detail": "Original Prompt"}}}
            
            return answer
            
            
            
    if request.method == 'GET':
        get_type = request.args.get('type')
        
        
        if get_type == "model_list" :
            #userid = request.args.get('ID')
            sql = "SELECT model_name FROM model_list"
            cur.execute(sql)
            rows = cur.fetchall()
            model_list = [list(rows[x]) for x in range(len(rows))]
            print(model_list)
            return model_list
        
        
        elif get_type == "item_detail":
            item_id = request.args.get('item_id')
            sql = "SELECT * FROM item where item_id = \"" +item_id + "\";"
            cur.execute(sql)
            rows = cur.fetchall()
            rows = [list(rows[x]) for x in range(len(rows))]
            item_info_list = sum(rows, [])
            item_info = dict(zip(item_list,item_info_list))
            sql = "update item set viewcount = viewcount +1 where item_id = %s;"
            cur.execute(sql,item_id)
            con.commit()
            return item_info
        
        elif get_type == "hot_list":
            sql = "SELECT * FROM item where is_sell = 1"
            cur.execute(sql)
            rows = cur.fetchall()
            rows = [list(rows[x]) for x in range(len(rows))]
            rows = [dict(zip(item_list,row)) for row in rows]
            hot_list = sorted(rows, key=itemgetter('viewcount'), reverse=True)
            return hot_list
            
        elif get_type == "new_list":
            sql = "SELECT * FROM item where is_sell = 1"
            cur.execute(sql)
            rows = cur.fetchall()
            rows = [list(rows[x]) for x in range(len(rows))]
            rows = [dict(zip(item_list,row)) for row in rows]
            new_list = sorted(rows, key=itemgetter('create_time'), reverse=True)
            return new_list
        
        elif get_type == "my_list":
            ID = request.args.get('ID')
            sql = "SELECT * FROM item where ID = %s"
            cur.execute(sql,ID)
            rows = cur.fetchall()
            rows = [list(rows[x]) for x in range(len(rows))]
            rows = [dict(zip(item_list,row)) for row in rows]
            return rows
            
            
        elif get_type == "search_list":
            sql = "SELECT * FROM item where is_sell = 1"
            cur.execute(sql)
            rows = cur.fetchall()
            rows = [list(rows[x]) for x in range(len(rows))]
            rows = [dict(zip(item_list,row)) for row in rows]
            return rows
        
        elif get_type == "get_category":
            sql = "SELECT category FROM item where is_sell = 1"
            cur.execute(sql)
            rows = cur.fetchall()
            rows = [list(rows[x]) for x in range(len(rows))]
            categories = sum(rows, [])
            categories = list(set(categories))
            return categories
        
        elif get_type == "download":
            item_id = request.args.get('item_id')
            sql = "update item set downloadcount = downloadcount +1 where item_id = %s;"
            cur.execute(sql,item_id)
            con.commit()
            return "OK"

        
@app.route('/static/<path:path>')
def send_report(path):
    print(path)
    return send_from_directory('static', path)


# @app.route("/login2", methods=["POST"])  
# def login():
#     con = pymysql.connect(host='127.0.0.1',port=3306, user='root',db='free3d', charset='utf8') 
#     cur = con.cursor()
    
#     ID = request.form["ID"]
#     password = request.form["password"]
#     sql = "Select ID from user"
#     cur.execute(sql)
#     rows = cur.fetchall()
#     rows = [list(rows[x]) for x in range(len(rows))]
#     IDS = sum(rows, [])
#     print('IDS', IDS)
#     if ID in IDS:
#         sql = "Select password from user where ID = \"" +ID+ "\";"
#         cur.execute(sql)
#         rows = cur.fetchall()
#         rows = [list(rows[x]) for x in range(len(rows))]
#         pass_db = rows[0][0]
#         if pass_db == password:
#             print('jwt')
#             # JWT 생성
#             access_token = create_access_token(identity=ID)
#             response = make_response(jsonify({'access_token': access_token, 'email': ID}))
#             response.set_cookie('jwt_token', access_token, httponly=True)
#             return response
#         else:
#             return jsonify({'msg': '아이디 또는 비밀번호가 일치하지 않습니다.'}), 401
#     else:
#         return jsonify({'msg': '아이디 또는 비밀번호를 확인하세요.'}), 401



@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify({'msg': f'{current_user}님, 인증에 성공하셨습니다!'}), 200

@app.route('/token', methods=['GET'])
@jwt_required()
def protected2():
    username = request.cookies.get('access_token')
    current_user = get_jwt_identity()
    access_token = get_jwt()
    response = make_response(jsonify({'access_token': access_token, 'email': current_user}))
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = '5000',debug=True)
            