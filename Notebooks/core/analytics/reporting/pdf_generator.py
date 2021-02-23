import os
import shutil
import calendar
from datetime import datetime
from xhtml2pdf import pisa
from jinja2 import Environment, FileSystemLoader
from PyPDF2 import PdfFileMerger
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import subprocess


class PDFGenerator:
    _file_base = None
    _file_templates = None
    _idf_logo = None

    _env = None

    _page_num = None

    def __init__(self):
        self._file_base = os.path.dirname(os.path.abspath(__file__))
        self._file_templates = self._file_base + '\\templates\\'
        self._idf_logo = f"{self._file_templates}IdentiFlight_Logo.png"

        self._env = Environment(loader=FileSystemLoader(self._file_templates))

        self._page_num = 0

    def __del__(self):
        """ When this class is deleted we want to remove all the temporary files used"""
        for page in range(self._page_num):
            os.remove(f"{self._file_templates}{page}.pdf")

        if os.path.exists(f'{self._file_templates}temp_fig.png'):
            os.remove(f'{self._file_templates}temp_fig.png')

    def _save_html_to_pdf(self, template, template_vars):
        """
        Given template html and the dictionary of variable it will create a single pdf page
        @param template: template html as string
        @param template_vars: dictionary of variables and values to insert into html
        @return: if conversion is successful, as bool
        """
        result_file = open(f"{self._file_templates}{self._page_num}.pdf", "w+b")
        pisa_status = pisa.CreatePDF(template.render(template_vars), result_file)
        result_file.close()
        return not pisa_status.err

    def _generate_approval_page(self, config):
        """
        Since the approval page has editable boxes we can not use the html approach.
        So instead we will construct it with report lab
        @param config: config name of model
        """
        if config is None:
            config = ''
        c = canvas.Canvas(filename=f"{self._file_templates}{self._page_num}.pdf")
        # Add idf logo
        c.drawImage(self._idf_logo, x=36, y=693, width=250, height=45)
        # Page title
        c.setFont("Helvetica", 28)
        c.drawCentredString(300, 650, 'viNet Model Approval')
        current_y = 635
        current_x = 20
        box_x = current_x + 130
        c.setLineWidth(1)
        c.line(20, current_y, 612 - 20, current_y)

        # Create form for editable fields
        form = c.acroForm
        c.setFont("Helvetica", 16)
        current_y -= 35
        c.drawString(current_x, current_y, 'Model name:')
        current_y -= 15
        form.textfield(name='mname', tooltip='Model name',
                       x=box_x, y=current_y, borderStyle='inset',
                       borderColor=colors.gray, fillColor=colors.floralwhite,
                       width=400, fontSize=14, value=config,
                       textColor=colors.royalblue, forceBorder=True)
        current_y -= 35
        c.drawString(current_x, current_y, 'Comments:')
        current_y -= 45
        form.textfield(name='comment', tooltip='Notes or Comments',
                       x=box_x, y=current_y, borderStyle='inset',
                       borderColor=colors.gray, fillColor=colors.floralwhite,
                       width=400, height=60, fontSize=14,
                       textColor=colors.royalblue, forceBorder=True, value='\n')
        current_y -= 35 + 100

        # Add signatures
        c.drawString(current_x, current_y, 'Signatures:')
        current_y -= 15

        for i in range(4):
            form.textfield(name=f'sig', tooltip='Signature',
                           x=box_x + 12, y=current_y, borderStyle='inset', fontSize=14, textColor=colors.royalblue,
                           width=388, fillColor=colors.floralwhite, borderColor=colors.floralwhite, borderWidth=0)
            c.line(box_x, current_y - 1, box_x + 400, current_y)
            c.drawString(x=box_x + 1, y=current_y, text='x')
            current_y -= 45
        c.save()
        return os.path.exists(f"{self._file_templates}{self._page_num}.pdf")

    def create_page(self, page_type, config=None, tag=None, title=None, plot_img_loc=None, plot_fig=None, custom=None,
                    plot_dpi=800):
        """
        will create a pdf of a single page for later use. If page can not be created then it will skip page. and print
        details
        @param page_type: specifies the template to use, title, plot, approval, appendix or custom
        @param config: config name, needed for title and approval
        @param tag: verification dataset tag, needed for title
        @param title: title for title page
        @param plot_img_loc: image location of plot, needed for plot
        @param plot_fig: can be used in place of the image location
        @param custom: if page_type is 'custom' then you can specify a custom pdf page
        @param plot_dpi: dpi for plot if plot_fig is provided
        """
        if page_type == 'title':
            if not config and not tag:
                print("Could not create title page. Make sure both a config and tag are provided to create_page().")
                return
            if title is None:
                title = "viNet Performance report"
            now = datetime.now()
            template = self._env.get_template(r"title_page.html")
            template_vars = {"config": config, "tag": tag, "page_num": self._page_num, "title": title,
                             "img_loc": self._idf_logo,
                             "date": f"{calendar.month_name[now.month]} {now.day}, {now.year}"}
            assert self._save_html_to_pdf(template, template_vars)

        elif page_type == 'plot':
            if not (plot_img_loc or plot_fig):
                print("Could not create plot page. Make sure plot_img_loc or plot_fig is provided to create_page().")
                return
            if plot_fig:
                plot_img_loc = f'{self._file_templates}temp_fig.png'
                plot_fig.savefig(plot_img_loc, dpi=plot_dpi)
            template = self._env.get_template(r"plot_template.html")
            template_vars = {"img_loc": self._idf_logo,
                             "plot_img": f"{plot_img_loc}", "page_num": self._page_num}
            assert self._save_html_to_pdf(template, template_vars)
            if plot_fig:
                os.remove(plot_img_loc)

        elif page_type == 'approval':
            assert self._generate_approval_page(config)

        elif page_type == 'appendix':
            # The appendix is created by using the pre-compiled pdf. This is done as the python
            # libraries being used don't like latex. might be smart to just use a latex generator
            shutil.copy(f"{self._file_templates}appendix.pdf",
                        f"{self._file_templates}{self._page_num}.pdf")

        elif page_type == 'custom':
            if not custom:
                print("Could not create custom page. Make sure custom is provided to create_page().")
                return
            shutil.copy(custom,
                        f"{self._file_templates}{self._page_num}.pdf")
        else:
            raise BaseException(f"Not a valid page type: {page_type}")

        self._page_num += 1

    def save_pdf(self, save_location, open_pdf=False):
        """
        Saves all created pdfs as a single pdf
        @param open_pdf:
        @param save_location: where to save pdf.
        """
        pdf_merger = PdfFileMerger()
        for page in range(self._page_num):
            pdf_merger.append(f"{self._file_templates}{page}.pdf", import_bookmarks=False)
        pdf_merger.write(save_location)
        pdf_merger.close()

        if open_pdf:
            subprocess.Popen(save_location, shell=True)

    def generate_from_plots(self, save_loc, config, tag, plots, title=None, approval=True, appendix=True):
        """
        Automatic report creation from a list of plot figures or image locations
        @param save_loc: save location for pdf
        @param config: config name
        @param tag: tag used
        @param plots: list of plots either as figures or image file locations
        @param title: title for title page
        @param approval: include approval page
        @param appendix: include appendix page
        """
        self.create_page('title', config=config, tag=tag, title=title)
        for plot in plots:
            if isinstance(plot, str):
                self.create_page('plot', plot_img_loc=plot)
            elif plot is not None:
                self.create_page('plot', plot_fig=plot)
        if approval:
            self.create_page('approval', config=config)
        if appendix:
            self.create_page('appendix')
        self.save_pdf(save_location=save_loc, open_pdf=True)
