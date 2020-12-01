import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.font_manager as font_manager
import argparse
import os
import csv

parser = argparse.ArgumentParser(description='Options for td visualization.')
parser.add_argument('--frame', type=int, default=1, help='frame serious for plotting')
parser.add_argument('--csv_path', type=str, help='path to where csvs are placed', default="")
args = parser.parse_args()

def plot_img(name, x1, y1, x2, y2):
    fig, ax = plt.subplots(1,dpi=300,figsize=(5,2))
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.plot(x1,y1,'b',linewidth=1)
    ax.plot(x2,y2,'r',linewidth=1)
    
    plt.xlim(x1[0],x1[-1])
    plt.xlabel('Iteration')
    plt.ylabel(name)
    if name == "Norm_Translation/Mean_Depth":
        plt.legend(['Pred','GT'], loc='upper center', bbox_to_anchor=(0.9,0.5))
        plt.savefig('t_d_{}.png'.format(args.frame),bbox_inches='tight')
    else:
        plt.legend(['Pred','GT'], loc='upper center', bbox_to_anchor=(0.9,0.8))
        plt.savefig('{}_{}.png'.format(name, args.frame),bbox_inches='tight')
    plt.close()
    

def main():
    data = {}
    keys = ['gt_d', 'gt_t', "gt_t_d", "pred_d", "pred_t", "pred_t_d"]
    tags = ["d_{}".format(args.frame), "t_{}".format(args.frame), "t_d_{}".format(args.frame), "d_{}".format(args.frame), "t_{}".format(args.frame), "t_d_{}".format(args.frame)]
    for i,key i        data_path = os.path.join(args.csv_path,"run-{}_{}-tag-{}.csv".format(tags[i],key,tags[i]))
:n enumerate(keys)
        with open(data_path,'r') as f:
            reader = csv.reader(f)
            data_key = list(reader)
            data_key_arr = np.array(data_key)
            data[(key,'x')] = data_key_arr[1:,1].astype(np.float)
            data[(key,'y')] = data_key_arr[1:,2].astype(np.float)
    plot_img("Mean_Depth",data[('pred_d','x')], data[('pred_d','y')], data[('gt_d','x')], data[('gt_d','y')])
    plot_img("Norm_Translation",data[('pred_t','x')], data[('pred_t','y')], data[('gt_t','x')], data[('gt_t','y')])
    plot_img("Norm_Translation/Mean_Depth",data[('pred_t_d','x')], data[('pred_t_d','y')], data[('gt_t_d','x')], data[('gt_t_d','y')])


if __name__ == "__main__":
    main()
    


'''    
f_ids = range(697)

mean_scale = np.load('mean_scale28.npy')

'''