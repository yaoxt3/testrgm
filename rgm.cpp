#include <iostream>
#include <fstream>
#include <stack>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//aaaaaa
int MAX = 500;
int max_set_num = 50;
int setnum = 1; //set order number
const int max_shift = 2;
const int max_ptemp = 10000;

struct Pointset
{
    int x;
    int y;
    int flag;
    int pixel;
    Pointset(){
        x = 0;
        y = 0;
        flag = 0;
        pixel = -1;
    }
    Pointset(int x1, int y1){
        x = x1;
        y = y1;
        flag = 0;
        pixel = -1;
    }
    Pointset operator + (const Pointset& a)const{
        return Pointset(x+a.x,y+a.y); 
    }
};

// store the same flag point in nPointset.
struct nPointset
{
    vector< Pointset> pset;
    int num;
    int flag;
    nPointset(){
        num = 0;
        flag = 0;
        pset.clear();
    }
};

// moment invariant
struct MomentInv
{
    int flag;
    int mwidth;
    int mhigh;
    double hu[7]; // seven eignevalue

};

Pointset *pointshift; // search path
Pointset **points;    // point
Pointset *ptemp;
nPointset *v_nPointset; // store all point set in v_nPointset. 
MomentInv *momentinv;
void construct_shift();
int find_flag(int,int,int,int);
void RGM_function(Mat);
void MomentInvariant(int,int);

// this function is used to construct search path
void construct_shift()
{
    int maxnum = (2*max_shift+1)*(2*max_shift+1)-1;
    pointshift = new Pointset[maxnum];
    int cut = 0;
    for(int i=-max_shift;i<max_shift+1;i++){
        for(int j=-max_shift;j<max_shift+1;j++){
            if(i==0&&j==0) continue;
            pointshift[cut++] = Pointset(i,j);
        }
    }
}

int find_flag(int x,int y,int row,int col)
{
    int count = 0;
    int maxnum = (2*max_shift+1)*(2*max_shift+1)-1;
    stack<Pointset> seed;
    seed.push(points[x][y]);
    ptemp[count++] = seed.top();
    while(!seed.empty()){
        Pointset top = seed.top();
        points[top.x][top.y].flag = setnum;
        /* ptemp[count] = top; */
        /* count++; // statistics the point number */
        seed.pop();
        for(int i=0; i<maxnum; i++){
            Pointset p = top + pointshift[i];
            if(p.x>=0 && p.x<row && p.y>=0 && p.y<col && points[p.x][p.y].pixel==0 && points[p.x][p.y].flag==0){
                seed.push(points[p.x][p.y]);
            }
        }
    }
    // if point number smaller than 4, ignore this set
    /* if(count>4){ */
    /*     for(int i=0;i<count;i++){ */
    /*         points[ptemp[i].x][ptemp[i].y].flag = setnum; */
    /*     } */
    /*     setnum++; */
    /* } */
    /* else{ */
    /*     for(int i=0;i<count;i++){ */
    /*         points[ptemp[i].x][ptemp[i].y].flag = 0; */
    /*     } */
    /* } */
    setnum++;
    return 1;    
}

void RGM_function(Mat image)
{
    construct_shift();
    ptemp = new Pointset[max_ptemp];
    points = new Pointset*[MAX];
    for(int i=0; i<MAX;i++){
        points[i] = new Pointset[MAX];
    }
    int rows = image.rows;
    int cols = image.cols;
    for(int i=0; i<rows; i++){
        uchar *data = image.ptr<uchar>(i);
        for(int j=0; j<cols; j++){
            points[i][j].x = i;
            points[i][j].y = j;
            points[i][j].pixel = (int)data[j]/255;
        }
    }
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            if(points[i][j].pixel==0 && points[i][j].flag==0){
                find_flag(i,j,rows,cols);
            }
        }
    }
    MomentInvariant(rows,cols);

}

void MomentInvariant(int rows,int cols)
{
    v_nPointset = new nPointset[setnum];
    /* ofstream outfile("./out.txt"); */
    /* ofstream outfile1("./out1.txt"); */
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            /* outfile << points[i][j].flag; */
            /* outfile1 << points[i][j].pixel; */
            int flag0 = points[i][j].flag;
            if(flag0!=0){
                v_nPointset[flag0].pset.push_back(points[i][j]);
                v_nPointset[flag0].flag = flag0;
                v_nPointset[flag0].num++;
            }
        }
        /* outfile << endl; */
        /* outfile1 << endl; */
    }
    /* for(int i=0;i<setnum;i++){ */
    /*     cout << v_nPointset[i].flag << " " << v_nPointset[i].num << endl; */
    /* } */
    
    //construct subimage to calculate moment invariant elgnevaule
    momentinv = new MomentInv[setnum];
    for(int i=0; i<setnum; i++){
        int num = v_nPointset[i].num;
        int minwidth = 10000, maxwidth = -1;
        int minhigh = 10000, maxhigh = -1;
        for(int j=0; j<num; j++){
            int a = v_nPointset[i].pset[j].x;
            int b = v_nPointset[i].pset[j].y;
            if(b < minwidth)
                minwidth = b;
            if(b > maxwidth)
                maxwidth = b;
            if(a < minhigh)
                minhigh = a;
            if(a > maxhigh)
                maxhigh = a;
        }
        momentinv[i].mwidth = maxwidth - minwidth;
        momentinv[i].mhigh = maxhigh - minhigh;
    }

}

int main()
{
    /* Mat image = imread("../black.png"); */
    Mat image = imread("../pointcloud1.png");
    cvtColor(image,image,CV_BGR2GRAY);
    RGM_function(image);
    /* Mat image = imread("../pointcloud.png"); */
    /* cvtColor(image,image,CV_BGR2GRAY); */
    /* Mat result; */
    /* result=image.clone(); */
    /* result = 255 - result; */
    /* for(int i=0;i<result.rows;i++) */
    /* { */
    /*     uchar *data = result.ptr<uchar>(i); */
    /*     for(int j=0; j<result.cols;j++) */
    /*     { */
    /*         if((int)data[j]!=255) */
    /*             data[j] = 0; */
    /*     } */
    /* } */
    /* namedWindow("test",WINDOW_AUTOSIZE); */
    /* imshow("test",result); */
    /* imwrite("../pointcloud1.png",result); */
    
    /* threshold(image,result,80,150.0,CV_THRESH_BINARY); */
    /* namedWindow("test",WINDOW_AUTOSIZE); */
    /* imshow("test",result); */

    waitKey(0);

    return 0;
}
