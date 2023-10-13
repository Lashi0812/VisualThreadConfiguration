import random
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def run_index(gdim_x,gdim_y,bdim_x,bdim_y):
    Path.mkdir(Path("./build"),parents=True,exist_ok=True)
    subprocess.run(["nvcc","-o","./build/index","index.cu"])
    with open("out.csv","w") as out:
        subprocess.run(["./build/index",gdim_x,gdim_y,bdim_x,bdim_y],stdout=out)

def generate_random_hex_color(low=0,high=255):
  red = random.randint(low, high)
  green = random.randint(low, high)
  blue = random.randint(low, high)

  red_hex = hex(red)[2:].zfill(2)
  green_hex = hex(green)[2:].zfill(2)
  blue_hex = hex(blue)[2:].zfill(2)

  return f"#{red_hex}{green_hex}{blue_hex}"


def get_df(filepath):
  df = pd.read_csv(filepath,names=["blk_y","blk_x","gdim_y","gdim_x","warp","th_y","th_x","bdim_y","bdim_x"])
  gdim_x = df["gdim_x"][0]
  gdim_y = df["gdim_y"][0]
  bdim_x = df["bdim_x"][0]
  bdim_y = df["bdim_y"][0]
  return df.drop(columns=["gdim_x","gdim_y","bdim_x","bdim_y"]),(gdim_y,gdim_x),(bdim_y,bdim_x)


def save_grid_png(df,grid_dim,block_dim,out_filename):
  # Define the size of the grid
  grid_size_y = grid_dim[0] * block_dim[0]
  grid_size_x = grid_dim[1] * block_dim[1]

  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(grid_size_x,grid_size_y))

  # (0, 'blk_y') (1, 'blk_x') (2, 'warp') (3, 'th_y') (4, 'th_x') 
  for idx,blk in df.groupby(["blk_y","blk_x"]):
      blk_y = idx[0] * block_dim[0]
      blk_x = idx[1] * block_dim[1]
      
      ax.add_patch(plt.Rectangle((blk_x,blk_y),block_dim[1],block_dim[0],fc=generate_random_hex_color(),ec="black"))    
      warp_colors = [generate_random_hex_color() for _ in range(blk["warp"].max()+1)]
      for i in blk.values:        
          rx = blk_x + i[4] 
          ry = blk_y + i[3] 
          ax.add_patch(plt.Rectangle((rx,ry), 0.9, 0.9,ec="black",fc=warp_colors[i[2]],alpha=0.7))
          cx = rx + 0.9/2.0
          cy = ry + 0.9/2.0
          ax.annotate((i[3],i[4]), (cx, cy), color='w', weight='bold', 
                  fontsize=10, ha='center', va='center')

  # Set the aspect ratio to equal so squares are square
  ax.set_aspect('equal')
  ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
  
  # Set axis limits
  ax.set_xlim(0, grid_size_y)
  ax.set_ylim(0, grid_size_x)

  # Display the grid
  plt.gca().invert_yaxis()
  Path.mkdir(Path(out_filename).parent,parents=True,exist_ok=True)
  plt.savefig(out_filename, bbox_inches='tight', pad_inches=0.1)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='index',
                    description='To Visualize the thread Organization with warp')
    parser.add_argument('-gx', '--grid_dim_x',help='x dim of grid',default="2")
    parser.add_argument('-gy', '--grid_dim_y',help='y dim of grid',default="2")
    parser.add_argument('-bx', '--block_dim_x',help='x dim of block',default="8")
    parser.add_argument('-by', '--block_dim_y',help='y dim of block',default="8")
    parser.add_argument('-o', '--out_filename',help='out filepath for image ')
    args = parser.parse_args()
    
    if args.out_filename is None:
        args.out_filename = f"./asset/out_{args.grid_dim_x}x{args.grid_dim_x}_{args.block_dim_x}x{args.block_dim_x}.png"
        

    run_index(args.grid_dim_x,args.grid_dim_y,args.block_dim_x,args.block_dim_y)
    df,grid_dim,block_dim = get_df("out.csv")
    save_grid_png(df,grid_dim,block_dim,args.out_filename)