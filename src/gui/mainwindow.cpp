#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    mode = std::string("2d (high performance)");
    collisions = false;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_commandLinkButton_clicked()
{
    fileName = QFileDialog::getOpenFileName(this, tr("Open Input File"), "./");
}

void MainWindow::on_comboBox_currentIndexChanged(const QString &arg1)
{
    mode = arg1.toStdString();
    std::cout << mode;
}

void MainWindow::on_comboBox_2_currentIndexChanged(const QString &arg1)
{
    collisions = false;
}

void MainWindow::on_pushButton_clicked()
{
    QProcess p;
    QStringList args;

    args.push_back(fileName);
    if(mode == "2d") {
		p.start("./nbody/2d", args);
	}
    else if(mode == "3d") {
        p.start("./nbody/3dp", args);
	}
	else {
		args.push_back("1");
		p.start("./nbody/2d", args);
	}
    p.waitForFinished(-1);

}
