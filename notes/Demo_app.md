# running demo app on server

nvidia-smi # make sure GPU is free
tmux new -s demo_ui
conda activate toxicsql
python src/demo_app.py

# On your LAPTOP — open the tunnel (same pattern as your Moodle Jupyter setup)

ssh -N -L 7860:localhost:7860 a.eshpulatov@oth-aw.de@ki-030.oth-aw.de

# then open http://localhost:7860 in your browser

# connecting db from local computer

postgresql+psycopg2 :// you : pass @ localhost : 5432 / dummy_shop
└─ engine + driver ─┘ │ │ │ │ └─ database name
user password host port

user / password — whatever you set when you made the local DB (often your username, or postgres / root).

host = localhost, port = 5432 (Postgres) or 3306 (MySQL).

database name — list them with psql -l (Postgres) or mysql -e "SHOW DATABASES;" (MySQL).

# for Mysql engine

MySQL is identical with mysql+pymysql://… and port 3306.

# testing on laptop first

import sqlalchemy as sa
e = sa.create_engine("postgresql+psycopg2://you:pass@localhost:5432/dummy_shop")
print(e.connect().execute(sa.text("SELECT 1")).fetchone())   # prints (1,) if it works

# Path A — reverse SSH tunnel (least work, recommended for testing now)

Keep the DBs on your laptop and "lend" them to DC1.07 for the session. One SSH command does both your Gradio tunnel and
the DB tunnel:

# from your LAPTOP — keep this open during the demo

ssh -L 7860:localhost:7860 \
-R 15432:localhost:5432 \
-R 13306:localhost:3306 \
a.eshpulatov@oth-aw.de@ki-030.oth-aw.de

-L 7860… → brings the Gradio UI to your laptop (as before).
-R 15432:localhost:5432 → makes DC1.07's port 15432 reach your laptop's Postgres (5432). (Using 15432/13306 instead of
the real ports avoids clashes with other students on the shared machine.)

Then in the Gradio connection-string box, you point at the tunnel port (from DC1.07's view):    
postgresql+psycopg2://you:pass@localhost:15432/dummy_shop
mysql+pymysql://root:pass@localhost:13306/dummy_blog

# creating the db and loading it
export PGPASSWORD=postgres   # password set as a postgres
cd ~/Desktop/OTH MAI/5-semester/AISP/db_backups/         # <-- change to wherever your pagila files are

createdb -h localhost -U postgres pagila
psql -h localhost -U postgres -d pagila -f pagila-schema.sql
psql -h localhost -U postgres -d pagila -f pagila-data.sql

# verify and look at locally
psql -h localhost -U postgres -d pagila -c "\dt"                                   # list all tables
psql -h localhost -U postgres -d pagila -c "SELECT staff_id, username, password FROM staff;"

# password example for pagila db 
postgresql+psycopg2://postgres:demo123@localhost:5432/pagila

# opening the tunnel and running demo after runnig on server tunneling on laptop
ssh -L 7860:localhost:7860 -R 15432:localhost:5432 a.eshpulatov@oth-aw.de@ki-030.oth-aw.de