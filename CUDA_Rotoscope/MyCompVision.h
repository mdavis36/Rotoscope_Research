#ifndef MYCOMPVISION_H
#define MYCOMPVISION_H



struct momentData
{
      int m_00; int u_00;
      int m_10; int u_10;
      int m_01; int u_01;
      int m_11; int u_11;
      int m_20; int u_20;
      int m_02; int u_02;
      int m_22; int u_22;
      int c_x;  int c_y;
};

struct PCAData
{
      float lam_1, lam_2;
      float theta;
      float maj_len;
      float min_len;
      float ecc;
};

bool inBounds(int w, int h, int x, int y)
{
      return x< w && x >= 0 && y < h && y >= 0 ? true : false;
}

void histogram_equalization(cv::Mat img, cv::Mat &out)
{
      std::cout << "Histogram Equalization" << std::endl;

      int w_size = 3;

      cv::Mat img_gray;

      float hist[256];
      float cdf[256];
      for (int i = 0; i < 256; i++)
      {
            hist[i] = 0.0f;
            cdf[i] = 0.0f;
      }

      cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
      out = cv::Mat(img.size(), CV_8UC1);

      //Populate Histogram
      for (int i = 0; i < img.cols; i++)
      {
            for (int j = 0; j < img.rows; j++)
            {
                  hist[(int)img_gray.at<uchar>(j,i)]++;
            }
      }

      for(int i = 0; i < 256; i++)
      {
            hist[i] /= (img.cols * img.rows);
      }

      //Populate CDF
      cdf[0] = hist[0];
      for (int i = 1; i < 256; i++)
      {
            cdf[i] = cdf[i-1] + hist[i];
      }

      //Populate Output
      for (int i = 0; i < img.cols; i++)
      {
            for (int j = 0; j < img.rows; j++)
            {
                  out.at<uchar>(j,i) = (255 * cdf[(int)img_gray.at<uchar>(j,i)]);
            }
      }
}

float ridleCalvard(cv::Mat img)
{
      float hist[256];
      float m_0[256];
      float m_1[256];
      for (int i = 0; i < 256; i++)
      {
            hist[i] = 0.0f;
      }

      //Populate Histogram
      for (int i = 0; i < img.cols; i++)
      {
            for (int j = 0; j < img.rows; j++)
            {
                  hist[(int)img.at<uchar>(j,i)]++;
            }
      }

      //Generate m_0
      m_0[0] = hist[0];
      m_1[0] = 0;
      for (int i = 1; i < 256; i++)
      {
            m_0[i] = m_0[i-1] + hist[i];
            m_1[i] = m_1[i-1] + (hist[i] * i);
      }

      int t = 127;
      int next_t = t;
      float u_b, u_f;


      int j = 0;
      do
      {
            t = next_t;
            //std::cout << "t = " << next_t << std::endl;
            u_b = (float) m_1[t] / m_0[t];
            u_f = (float) (m_1[255] - m_1[t]) / (m_0[255] - m_0[t]);
            next_t = (u_b + u_f) / 2.0f;
            j++;
      }while(abs(t - next_t) > 0.01f);

      return next_t;
}

void threashold(cv::Mat img, cv::Mat &out)
{
      std::cout << "Thresholding Ridle-Calvard" << std::endl;

      cv::Mat img_gray;
      cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
      cv::imshow("Input", img);
      out = cv::Mat(img.size(), CV_8UC1);


      float t = ridleCalvard(img_gray);
      for (int i = 0; i < img.cols; i++)
      {
            for (int j = 0; j < img.rows; j++)
            {
                  if (img_gray.at<uchar>(j,i) > t)
                        out.at<uchar>(j,i) = 255;
                  else
                        out.at<uchar>(j,i) = 0;
            }
      }
}

void threashold(cv::Mat img, cv::Mat &out, float t)
{
      std::cout << "Thresholding Ridle-Calvard" << std::endl;

      cv::Mat img_gray;
      cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
      cv::imshow("Input", img);
      out = cv::Mat(img.size(), CV_8UC1);

      for (int i = 0; i < img.cols; i++)
      {
            for (int j = 0; j < img.rows; j++)
            {
                  if (img_gray.at<uchar>(j,i) > t)
                        out.at<uchar>(j,i) = 255;
                  else
                        out.at<uchar>(j,i) = 0;
            }
      }
}

void floodFillCheckNeighbour(std::vector<cv::Point> &f, cv::Point p, cv::Mat img, cv::Vec3b s)
{
      if (img.at<cv::Vec3b>(p) == s)
      {
            bool exists = false;
            for (int i = 0; i < f.size(); i++)
            {
                  if (f[i] == p)
                  {
                        exists = true;
                        break;
                  }
            }
            if (!exists) f.push_back(p);
      }
}

void floodFillCheckNeighbour(std::vector<cv::Point> &f, cv::Point p, cv::Mat img, unsigned char s)
{
      if (img.at<unsigned char>(p) == s)
      {
            bool exists = false;
            for (int i = 0; i < f.size(); i++)
            {
                  if (f[i] == p)
                  {
                        exists = true;
                        break;
                  }
            }
            if (!exists) f.push_back(p);
      }
}

void floodFillFrontier(cv::Mat& img, cv::Vec3b c, int x, int y)
{
      std::cout << "Flood Fill Frontier" << std::endl;

      std::vector<cv::Point> frontier;
      cv::Point curr;

      frontier.push_back(cv::Point(x,y));

      int i = 0;

      cv::Vec3b seed = img.at<cv::Vec3b>(x,y);
      while (!frontier.empty())
      {
            curr = frontier.back();
            frontier.pop_back();
            floodFillCheckNeighbour(frontier, cv::Point(curr.x - 1, curr.y), img, seed);
            floodFillCheckNeighbour(frontier, cv::Point(curr.x + 1, curr.y), img, seed);
            floodFillCheckNeighbour(frontier, cv::Point(curr.x, curr.y + 1), img, seed);
            floodFillCheckNeighbour(frontier, cv::Point(curr.x, curr.y - 1), img, seed);
            img.at<cv::Vec3b>(curr) = c;
      }
}

void floodFillFrontier(cv::Mat img, cv::Mat& out, cv::Vec3b c, int x, int y)
{
      std::cout << "Flood Fill Frontier" << std::endl;

      std::vector<cv::Point> frontier;
      cv::Point curr;

      frontier.push_back(cv::Point(x,y));

      int i = 0;

      cv::Vec3b seed = img.at<cv::Vec3b>(x,y);
      while (!frontier.empty())
      {
            curr = frontier.back();
            frontier.pop_back();
            floodFillCheckNeighbour(frontier, cv::Point(curr.x - 1, curr.y), img, seed);
            floodFillCheckNeighbour(frontier, cv::Point(curr.x + 1, curr.y), img, seed);
            floodFillCheckNeighbour(frontier, cv::Point(curr.x, curr.y + 1), img, seed);
            floodFillCheckNeighbour(frontier, cv::Point(curr.x, curr.y - 1), img, seed);
            img.at<cv::Vec3b>(curr) = c;
            out.at<cv::Vec3b>(curr) = c;
      }
}

void floodFillFrontier(cv::Mat img, cv::Mat& out, unsigned char c, int x, int y)
{
      std::cout << "Flood Fill Frontier" << std::endl;

      std::vector<cv::Point> frontier;
      cv::Point curr;

      frontier.push_back(cv::Point(x,y));

      int i = 0;

      unsigned char seed = img.at<unsigned char>(x,y);
      while (!frontier.empty())
      {
            curr = frontier.back();
            frontier.pop_back();
            floodFillCheckNeighbour(frontier, cv::Point(curr.x - 1, curr.y), img, seed);
            floodFillCheckNeighbour(frontier, cv::Point(curr.x + 1, curr.y), img, seed);
            floodFillCheckNeighbour(frontier, cv::Point(curr.x, curr.y + 1), img, seed);
            floodFillCheckNeighbour(frontier, cv::Point(curr.x, curr.y - 1), img, seed);
            img.at<unsigned char>(curr) = c;
            out.at<unsigned char>(curr) = c;
      }
}

void floodFillRecursive(cv::Mat &img,  int r, int c, unsigned char seed, unsigned char color)
{
      int w = img.cols;
      int h = img.rows;
      if (img.at<uchar>(r,c) == seed)
      {
            img.at<uchar>(r,c) = color;
            if(inBounds(w,h,c,r+1)) floodFillRecursive(img, r + 1, c, seed, color);
            if(inBounds(w,h,c,r-1)) floodFillRecursive(img, r - 1, c, seed, color);
            if(inBounds(w,h,c+1,r)) floodFillRecursive(img, r, c + 1, seed, color);
            if(inBounds(w,h,c-1,r)) floodFillRecursive(img, r, c - 1, seed, color);
      }
}

void floodFillRecursive(cv::Mat &original, cv::Mat &img, cv::Mat &out, int x, int y, unsigned char seed, unsigned char color)
{
      int w = original.cols;
      int h = original.rows;
      if (img.at<uchar>(x,y) == seed)
      {
            out.at<uchar>(x,y) = color;
            img.at<uchar>(x,y) = color;
            if(inBounds(w,h,x+1,y)) floodFillRecursive(original, img, out, x + 1, y, seed, color);
            if(inBounds(w,h,x-1,y)) floodFillRecursive(original, img, out, x - 1, y, seed, color);
            if(inBounds(w,h,x,y+1)) floodFillRecursive(original, img, out, x, y + 1, seed, color);
            if(inBounds(w,h,x,y-1)) floodFillRecursive(original, img, out, x, y - 1, seed, color);
      }
}

void floodFillRecursive(cv::Mat img, cv::Mat &out, int x, int y, unsigned char seed, unsigned char color)
{
      cv::Mat og;
      img.copyTo(og);
      floodFillRecursive(img, og, out, x, y, seed, color);
}

void doubleThreshold(cv::Mat img, cv::Mat &out)
{
      float t_hi = ridleCalvard(img);
      float t_lo = t_hi /2;

      cv::Mat hi(img.size(), CV_8UC1);
      cv::Mat lo(img.size(), CV_8UC1);

      threashold(img, hi, t_hi);
      threashold(img, lo, t_lo);

      unsigned char val = 255;

      for (int i = 0; i < img.cols; i++)
      {
            for (int j = 0; j < img.rows; j++)
            {
                  if (hi.at<unsigned char>(j,i) == 255 &&
                      lo.at<unsigned char>(j,i) == 255 &&
                      out.at<unsigned char>(j,i) != 127)
                  {
                        floodFillRecursive(hi, out, j, i, 255, 127);
                  }
            }
      }
      for (int i = 0; i < img.cols; i++)
      {
            for (int j = 0; j < img.rows; j++)
            {
                  if (out.at<unsigned char>(j,i) == 127)
                        out.at<unsigned char>(j,i) = 255;
            }
      }


}

void binaryDilation(cv::Mat in, cv::Mat &out)
{
      cv::Mat temp;
      in.copyTo(temp);
      bool checkIntersect;
      for (int i = 0; i < in.cols; i++)
      {
            for (int j = 0; j < in.rows; j++)
            {
                  checkIntersect = false;
                  //Center
                  if (temp.at<unsigned char>(j, i) == 255) checkIntersect = true;

                  //Left
                  if (inBounds(in.cols, in.rows, i-1, j))
                  {
                        if (temp.at<unsigned char>(j, i-1) == 255) checkIntersect = true;
                  }
                  //Right
                  if (inBounds(in.cols, in.rows, i+1, j))
                  {
                        if (temp.at<unsigned char>(j, i+1) == 255) checkIntersect = true;
                  }
                  //Up
                  if (inBounds(in.cols, in.rows, i, j-1))
                  {
                        if (temp.at<unsigned char>(j-1, i) == 255) checkIntersect = true;
                  }
                  //Down
                  if (inBounds(in.cols, in.rows, i, j+1))
                  {
                        if (temp.at<unsigned char>(j+1, i) == 255) checkIntersect = true;
                  }

                  if (checkIntersect) out.at<unsigned char>(j,i) = 255;
            }
      }
}

void binaryErosion(cv::Mat in, cv::Mat &out)
{
      cv::Mat result = cv::Mat::zeros(in.size(), CV_8UC1);
      cv::Mat temp;
      in.copyTo(temp);
      bool checkIntersect;
      for (int i = 0; i < in.cols; i++)
      {
            for (int j = 0; j < in.rows; j++)
            {
                  checkIntersect = true;
                  //Center
                  if (temp.at<unsigned char>(j, i) != 255) checkIntersect = false;

                  //Left
                  if (inBounds(in.cols, in.rows, i-1, j))
                  {
                        if (temp.at<unsigned char>(j, i-1) != 255) checkIntersect = false;
                  }
                  //Right
                  if (inBounds(in.cols, in.rows, i+1, j))
                  {
                        if (temp.at<unsigned char>(j, i+1) != 255) checkIntersect = false;
                  }
                  //Up
                  if (inBounds(in.cols, in.rows, i, j-1))
                  {
                        if (temp.at<unsigned char>(j-1, i) != 255) checkIntersect = false;
                  }
                  //Down
                  if (inBounds(in.cols, in.rows, i, j+1))
                  {
                        if (temp.at<unsigned char>(j+1, i) != 255) checkIntersect = false;
                  }

                  if (checkIntersect) result.at<unsigned char>(j,i) = 255;
            }
      }
      out = result;
}

int connectedComponents(cv::Mat in, cv::Mat &out)
{
      cv::Mat result = cv::Mat(in.size(), CV_8UC1);
      in.copyTo(result);
      unsigned char nextLabel = 1;
      for (int j = 0; j < in.rows; j++)
      {
            for (int i = 0; i < in.cols; i++)
            {
                  if (result.at<uchar>(j,i) == 255)
                  {
                        floodFillRecursive(result, j, i, 255, nextLabel);
                        nextLabel ++;
                  }
            }
      }
      result.copyTo(out);
      return nextLabel - 1;
}

void regionProperties(cv::Mat in, int label, momentData &mData)
{
      mData.m_00 = 0; mData.u_00 = 0;
      mData.m_10 = 0; mData.u_10 = 0;
      mData.m_01 = 0; mData.u_01 = 0;
      mData.m_11 = 0; mData.u_11 = 0;
      mData.m_20 = 0; mData.u_20 = 0;
      mData.m_02 = 0; mData.u_02 = 0;
      mData.m_22 = 0; mData.u_22 = 0;
      mData.c_x = 0;  mData.c_y = 0;
      for (int j = 0; j < in.rows; j++)
      {
            for (int i = 0; i < in.cols; i++)
            {
                  if (in.at<uchar>(j,i) == label)
                  {
                        mData.m_00++;
                        mData.m_10 += j;
                        mData.m_01 += i;
                        mData.m_11 += i*j;
                        mData.m_20 += j*j;
                        mData.m_02 += i*i;
                  }
            }
      }
      mData.c_x = mData.m_10 / mData.m_00;
      mData.c_y = mData.m_01 / mData.m_00;

      mData.u_00 = mData.m_00;
      mData.u_11 = mData.m_11 - mData.c_x * mData.m_01;
      mData.u_20 = mData.m_20 - mData.c_x * mData.m_10;
      mData.u_02 = mData.m_02 - mData.c_y * mData.m_01;
}

void PCA(momentData m, PCAData &p)
{

      //std::cout << eigenVals.at<float>(0,0) << " " << eigenVals.at<float>(1,0) << std::endl;

      // double deter = ( ((m.u_20 - m.u_02) * (m.u_20 - m.u_02)) + 4 * (m.u_11 * m.u_11) );
      // p.lam_1 = (1 / (2 * m.u_00)) * (m.u_20 + m.u_02 + deter);
      // p.lam_2 = (1 / (2 * m.u_00)) * (m.u_20 + m.u_02 - deter);

      // Unfortunately I had to resort to using the OpenCV function to calculate the eigenvalues.
      // I kept getting overflow issues and didn't think that going into performing
      // array based mathematical operations was in the scope of this program for C++
      float u20_u00 = m.u_20 / m.u_00;
      float u02_u00 = m.u_02 / m.u_00;
      float u11_u00 = m.u_11 / m.u_00;

      cv::Mat covMat(cv::Size(2,2), CV_32FC1);
      cv::Mat eigenVals(cv::Size(2,0), CV_32FC1);
      covMat.at<float>(0,0) = u20_u00;
      covMat.at<float>(0,1) = u11_u00;
      covMat.at<float>(1,1) = u02_u00;
      covMat.at<float>(1,0) = u11_u00;

      cv::eigen(covMat, eigenVals);


      p.lam_1 = eigenVals.at<float>(0,0);
      p.lam_2 = eigenVals.at<float>(1,0);

      p.maj_len = 2 * sqrt(p.lam_1);
      p.min_len = 2 * sqrt(p.lam_2);
      p.theta = 0.5 * atan2(2 * m.u_11, m.u_20 - m.u_02);
      p.ecc = ((p.lam_1 - p.lam_2)/p.lam_1);
}

int getLeft(int dir){ return (dir + 3) % 4;}
int getRight(int dir){ return (dir + 1) % 4;}
cv::Point addp(cv::Point p1, cv::Point p2){ return cv::Point(p1.x + p2.x, p1.y + p2.y);}

void WallFollow(cv::Mat labelImg, cv::Mat outImg, int target, cv::Vec3b color)
{
      cv::Point dirs[4];
      dirs[0] = cv::Point(-1,0);
      dirs[2] = cv::Point(1,0);
      dirs[1] = cv::Point(0,1);
      dirs[3] = cv::Point(0,-1);

      int dir = 0;
      cv::Mat testimg = cv::Mat::zeros(labelImg.size(), CV_8UC1);

      for (int j = 0; j < labelImg.rows; j++)
      {
            for (int i = 0; i < labelImg.cols; i++)
            {
                  if (labelImg.at<uchar>(j,i) == target)
                  {
                        cv::Point currPos(i,j);
                        cv::Point initPos(i,j);
                        cv::Point temp = addp(currPos, dirs[dir]);

                        while (labelImg.at<uchar>(addp(currPos, dirs[dir])) == target)
                        {
                              dir = getRight(dir);
                              //temp = addp(currPos, dirs[dir]);
                        }
                        dir = getRight(dir);
                        std::cout <<"check"<< std::endl;
                        do{
                              outImg.at<cv::Vec3b>(currPos) = color;
                              //testimg.at<uchar>(currPos) = 255;

                              int ldir = getLeft(dir);
                              int rdir = getRight(dir);
                              int fronVal = (int)labelImg.at<uchar>(addp(currPos, dirs[dir]));
                              int leftVal = (int)labelImg.at<uchar>(addp(currPos, dirs[ldir]));
                              int rightVal = (int)labelImg.at<uchar>(addp(currPos, dirs[rdir]));

                              if (leftVal == target)
                              {
                                    dir = ldir;
                                    currPos = addp(currPos, dirs[dir]);
                              }
                              else if (fronVal != target)
                              {
                                    dir = rdir;
                              }
                              else
                              {
                                    currPos = addp(currPos, dirs[dir]);
                              }
                              //cv::namedWindow( "Test", cv::WINDOW_AUTOSIZE );
                              //cv::imshow("Test", outImg);
                              //cv::waitKey(0);

                        }while(currPos != initPos);
                        return;
                  }

            }
      }
}

int equivLabel(std::vector<int> eq, int index)
{
      //int i = 0;
      int currIndx = index;
      int nextIndx = eq[index];
      while (currIndx != nextIndx)
      {
            //std::cout << i << std::endl;
            //i++;
            currIndx = nextIndx;
            nextIndx = eq[currIndx];
      }
      return nextIndx;
}

void UnionFill(cv::Mat input, cv::Mat &out)
{
      out = cv::Mat(input.size(), CV_8UC1, cv::Scalar(-1));
      int rows = input.rows;
      int cols = input.cols;
      cv::Mat test(input.size(), CV_8UC1, cv::Scalar(0));

      std::vector<int> equiv1;
      equiv1.push_back(0);

      int currNum = 0;

      for (int j = 0; j < input.rows; j++)
      {
            for (int i = 0; i < input.cols; i++)
            {
                  if (j == 0 && i == 0) {
                        out.at<uchar>(j,i) = currNum;
                        i++;
                  }

                  int pix = input.at<uchar>(j,i);
                  if (pix == input.at<uchar>(j-1,i) && j-1 >= 0 &&
                      pix == input.at<uchar>(j,i-1) && i-1 >= 0)
                  {
                        if (input.at<uchar>(j,i-1)>input.at<uchar>(j-1,i))
                              equiv1[input.at<uchar>(j,i-1)]=out.at<uchar>(j-1,i);
                        else
                              equiv1[input.at<uchar>(j-1,i)]=out.at<uchar>(j,i-1);
                        //equiv1[out.at<uchar>(j-1,i)] = out.at<uchar>(j,i-1);
                        out.at<uchar>(j,i) = out.at<uchar>(j,i-1);
                  }
                  else if (pix == input.at<uchar>(j-1,i) && j-1 >= 0)
                  {
                        out.at<uchar>(j,i) = out.at<uchar>(j-1,i);
                  }
                  else if (pix == input.at<uchar>(j,i-1) && i-1 >= 0)
                  {
                        out.at<uchar>(j,i) = out.at<uchar>(j,i-1);
                  }
                  else
                  {
                        currNum++;
                        out.at<uchar>(j,i) = currNum;
                        equiv1.push_back(currNum);
                  }

                  test.at<uchar>(j,i) = 255;
                  //cv::imshow("test", test);
                  //cv::waitKey();
            }
      }
      for (int j = 0; j < input.rows; j++)
      {
            for (int i = 0; i < input.cols; i++)
            {
                  out.at<uchar>(j,i) = equivLabel(equiv1, out.at<uchar>(j,i));
            }
      }
}

#endif
