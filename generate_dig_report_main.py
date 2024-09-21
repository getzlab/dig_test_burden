import argparse

def generate_main_report(coding_report_path, utr3_report_path, utr5_report_path, promoter_report_path, output_path):
    # Read the content of the four DIG report HTML files
    with open(coding_report_path, 'r') as file:
        coding_content = file.read()
    with open(utr3_report_path, 'r') as file:
        utr3_content = file.read()
    with open(utr5_report_path, 'r') as file:
        utr5_content = file.read()
    with open(promoter_report_path, 'r') as file:
        promoters_content = file.read()

    # Template for main_report.html
    main_report_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width">
        <title>DIG Driver Report</title>
        <style>
            .report-section {{
                display: none;
            }}
            .active {{
                display: block;
            }}
            .navbar {{
                overflow: hidden;
                background-color: #333;
            }}
            .navbar a {{
                float: left;
                display: block;
                color: #f2f2f2;
                text-align: center;
                padding: 14px 16px;
                text-decoration: none;
            }}
            .navbar a:hover {{
                background-color: #ddd;
                color: black;
            }}
            .navbar a.active-link {{
                background-color: white;
                color: black;
            }}
            iframe {{
                width: 100%;
                height: 1000px;
                border: none;
            }}
        </style>
    </head>
    <body>
        <div class="navbar">
            <a href="#" onclick="showReport('coding', '{coding_report_path}')">Coding regions</a>
            <a href="#" onclick="showReport('promoters', '{promoter_report_path}')">Promoter regions</a>
            <a href="#" onclick="showReport('5-prime-utrs', '{utr5_report_path}')">5-prime UTRs</a>
            <a href="#" onclick="showReport('3-prime-utrs', '{utr3_report_path}')">3-prime UTRs</a>
        </div>

        <div id="coding" class="report-section active">
            <iframe id="report-frame" src="{coding_report_path}"></iframe>
        </div>

        <script>
            function showReport(reportId, reportUrl) {{
                var iframe = document.getElementById('report-frame');
                iframe.src = reportUrl;

                var links = document.querySelectorAll('.navbar a');
                links.forEach(link => link.classList.remove('active-link'));

                var activeLink = document.querySelector(`.navbar a[onclick*="${{reportId}}"]`);
                activeLink.classList.add('active-link');
            }}
        </script>
    </body>
    </html>
    """

    # Save the generated main_report.html
    with open(output_path, 'w') as file:
        file.write(main_report_template)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate main DIG report.")
    parser.add_argument("coding_report_path", type=str, help="Path to the DIG report HTML file for coding regions.")
    parser.add_argument("utr3_report_path", type=str, help="Path to the DIG report HTML file for 3-prime UTRs.")
    parser.add_argument("utr5_report_path", type=str, help="Path to the DIG report HTML file for 5-prime UTRs.")
    parser.add_argument("promoter_report_path", type=str, help="Path to the DIG report HTML file for promoter regions")
    parser.add_argument("output_path", type=str, help="Output path for the main DIG report HTML file.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # Generate the main report
    generate_main_report(args.coding_report_path, args.utr3_report_path, args.utr5_report_path, args.promoter_report_path, args.output_path)