#!/bin/bash

# IEEE Conference Paper Compilation Script
# This script compiles the LaTeX document to PDF

echo "🔧 Compiling IEEE Conference Paper..."
echo "📄 File: ieee_conference_paper.tex"
echo "🎯 Target: 12-page IEEE format document"
echo "================================================"

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install LaTeX distribution (MacTeX, TeX Live, or MiKTeX)"
    echo "💡 For macOS: brew install --cask mactex"
    echo "💡 For Ubuntu: sudo apt-get install texlive-full"
    echo "💡 For Windows: Download MiKTeX from miktex.org"
    exit 1
fi

# Compile the document (run twice for proper references)
echo "🔄 First compilation pass..."
pdflatex -interaction=nonstopmode ieee_conference_paper.tex > compilation.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ First pass completed successfully"
else
    echo "❌ First pass failed. Check compilation.log for errors"
    tail -20 compilation.log
    exit 1
fi

echo "🔄 Second compilation pass..."
pdflatex -interaction=nonstopmode ieee_conference_paper.tex >> compilation.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Second pass completed successfully"
    echo "🎉 PDF generated: ieee_conference_paper.pdf"
    
    # Check if PDF was actually created
    if [ -f "ieee_conference_paper.pdf" ]; then
        echo "📊 Document statistics:"
        echo "   📄 Pages: $(pdfinfo ieee_conference_paper.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo 'Unknown')"
        echo "   📏 Size: $(ls -lh ieee_conference_paper.pdf | awk '{print $5}')"
        echo "   📅 Created: $(date)"
        echo ""
        echo "🚀 Your IEEE conference paper is ready!"
        echo "📂 Location: $(pwd)/ieee_conference_paper.pdf"
    else
        echo "❌ PDF not found despite successful compilation"
        exit 1
    fi
else
    echo "❌ Second pass failed. Check compilation.log for errors"
    tail -20 compilation.log
    exit 1
fi

# Clean up auxiliary files (optional)
read -p "🧹 Clean up auxiliary files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f *.aux *.log *.out *.toc *.synctex.gz
    echo "✅ Auxiliary files cleaned up"
fi

echo "================================================"
echo "📖 To view your paper: open ieee_conference_paper.pdf"
echo "🔄 To recompile: ./compile_paper.sh"
echo "🎯 Paper follows IEEE conference format standards"